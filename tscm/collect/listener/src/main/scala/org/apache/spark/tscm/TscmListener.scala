package org.apache.spark.tscm

import java.io.{BufferedWriter, File, FileWriter}

import scala.collection.mutable

import org.apache.spark.SparkConf
import org.apache.spark.executor.{ExecutorMetrics, TaskMetrics}
import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart
import org.apache.spark.storage.{BlockUpdatedInfo, RDDBlockId}

/**
 * Writes one JSON line per event to ${spark.tscm.output}/<application id>.jsonl.
 *
 * It lives under org.apache.spark because RDDInfo.scope, which holds the operator names that
 * stage signatures hash, is private[spark]. Nothing here changes the job.
 */
class TscmListener(conf: SparkConf) extends SparkListener {

  private val outputDir = conf.get("spark.tscm.output", "tscm-events")
  private val pending = mutable.ArrayBuffer.empty[String]
  private var writer: Option[BufferedWriter] = None

  // Per-stage state, keyed by (stage id, attempt).
  private val peakMemory = mutable.Map.empty[(Int, Int), Long].withDefaultValue(0L)
  // Bytes of cached RDD blocks in memory, per executor and per block.
  private val cached = mutable.Map.empty[String, mutable.Map[String, Long]]
  private var maxExecutorCache = 0L

  private def emit(line: String): Unit = synchronized {
    writer match {
      case Some(out) =>
        out.write(line)
        out.newLine()
      case None => pending += line
    }
  }

  override def onApplicationStart(event: SparkListenerApplicationStart): Unit = synchronized {
    val appId = event.appId.getOrElse(s"app-${event.time}")
    new File(outputDir).mkdirs()
    val out = new BufferedWriter(new FileWriter(new File(outputDir, s"$appId.jsonl")))
    writer = Some(out)
    pending.foreach { line =>
      out.write(line)
      out.newLine()
    }
    pending.clear()
    emit(Json.obj("event" -> "app_start", "time" -> event.time, "app_id" -> appId))
  }

  override def onApplicationEnd(event: SparkListenerApplicationEnd): Unit = synchronized {
    emit(Json.obj("event" -> "app_end", "time" -> event.time))
    writer.foreach(_.close())
    writer = None
  }

  override def onJobStart(event: SparkListenerJobStart): Unit = {
    val sqlId = Option(event.properties)
      .flatMap(p => Option(p.getProperty("spark.sql.execution.id")))
      .map(_.toLong)
    emit(Json.obj(
      "event" -> "job",
      "job_id" -> event.jobId,
      "stage_ids" -> event.stageIds,
      "sql_execution_id" -> sqlId))
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = event match {
    case start: SparkListenerSQLExecutionStart =>
      emit(Json.obj(
        "event" -> "sql",
        "execution_id" -> start.executionId,
        "physical_plan" -> start.physicalPlanDescription))
    case _ =>
  }

  private def recordPeak(key: (Int, Int), metrics: ExecutorMetrics): Unit = synchronized {
    if (metrics != null) {
      val used = Seq("JVMHeapMemory", "ProcessTreeJVMRSSMemory")
        .map(metrics.getMetricValue).max
      val onHeap = metrics.getMetricValue("OnHeapExecutionMemory") +
        metrics.getMetricValue("OnHeapStorageMemory")
      peakMemory(key) = math.max(peakMemory(key), math.max(used, onHeap))
    }
  }

  override def onExecutorMetricsUpdate(event: SparkListenerExecutorMetricsUpdate): Unit = {
    event.executorUpdates.foreach { case (key, metrics) => recordPeak(key, metrics) }
  }

  override def onBlockUpdated(event: SparkListenerBlockUpdated): Unit = {
    val info: BlockUpdatedInfo = event.blockUpdatedInfo
    info.blockId match {
      case block: RDDBlockId => synchronized {
        val blocks = cached.getOrElseUpdate(info.blockManagerId.executorId, mutable.Map.empty)
        if (info.memSize > 0) blocks(block.name) = info.memSize else blocks.remove(block.name)
        maxExecutorCache = math.max(maxExecutorCache, blocks.values.sum)
      }
      case _ =>
    }
  }

  override def onTaskEnd(event: SparkListenerTaskEnd): Unit = {
    recordPeak((event.stageId, event.stageAttemptId), event.taskExecutorMetrics)
    val metrics: TaskMetrics = event.taskMetrics
    if (metrics != null) {
      emit(Json.obj(
        "event" -> "task",
        "stage_id" -> event.stageId,
        "stage_attempt" -> event.stageAttemptId,
        "task_id" -> event.taskInfo.taskId,
        "executor_id" -> event.taskInfo.executorId,
        "successful" -> event.taskInfo.successful,
        "duration_ms" -> event.taskInfo.duration,
        "executor_run_time_ms" -> metrics.executorRunTime,
        "executor_cpu_time_ns" -> metrics.executorCpuTime,
        "jvm_gc_time_ms" -> metrics.jvmGCTime,
        "peak_execution_memory" -> metrics.peakExecutionMemory,
        "memory_bytes_spilled" -> metrics.memoryBytesSpilled,
        "disk_bytes_spilled" -> metrics.diskBytesSpilled,
        "input_bytes" -> metrics.inputMetrics.bytesRead,
        "output_bytes" -> metrics.outputMetrics.bytesWritten,
        "shuffle_write_bytes" -> metrics.shuffleWriteMetrics.bytesWritten,
        "shuffle_local_bytes" -> metrics.shuffleReadMetrics.localBytesRead,
        "shuffle_remote_bytes" -> metrics.shuffleReadMetrics.remoteBytesRead))
    }
  }

  override def onStageCompleted(event: SparkListenerStageCompleted): Unit = synchronized {
    val info = event.stageInfo
    val key = (info.stageId, info.attemptNumber())
    val scopes = info.rddInfos.flatMap(_.scope.map(_.name)).distinct
    emit(Json.obj(
      "event" -> "stage",
      "stage_id" -> info.stageId,
      "attempt" -> info.attemptNumber(),
      "name" -> info.name,
      "num_tasks" -> info.numTasks,
      "parent_ids" -> info.parentIds,
      "rdd_scopes" -> scopes,
      "rdd_types" -> info.rddInfos.map(_.name).distinct,
      "call_site" -> info.rddInfos.lastOption.map(_.callSite).getOrElse(info.name),
      "reads_cache" -> info.rddInfos.exists(_.storageLevel.useMemory),
      "submission_time" -> info.submissionTime.getOrElse(0L),
      "completion_time" -> info.completionTime.getOrElse(0L),
      "failed" -> info.failureReason.isDefined,
      "storage_memory_max" -> maxExecutorCache,
      "executor_peak_memory" -> peakMemory(key)))
    peakMemory.remove(key)
  }
}

/** Minimal JSON writer; a listener must not bring dependencies onto the Spark classpath. */
private object Json {

  def obj(fields: (String, Any)*): String =
    fields.map { case (key, value) => s"${quote(key)}:${render(value)}" }.mkString("{", ",", "}")

  private def render(value: Any): String = value match {
    case null | None => "null"
    case Some(inner) => render(inner)
    case text: String => quote(text)
    case number: Long => number.toString
    case number: Int => number.toString
    case flag: Boolean => flag.toString
    case items: Iterable[_] => items.map(render).mkString("[", ",", "]")
    case other => quote(other.toString)
  }

  private def quote(text: String): String = {
    val out = new StringBuilder("\"")
    text.foreach {
      case '"' => out.append("\\\"")
      case '\\' => out.append("\\\\")
      case '\n' => out.append("\\n")
      case '\r' => out.append("\\r")
      case '\t' => out.append("\\t")
      case c if c < 0x20 => out.append(f"\\u${c.toInt}%04x")
      case c => out.append(c)
    }
    out.append('"').toString
  }
}
