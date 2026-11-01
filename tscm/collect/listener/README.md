# TSCM Spark listener

A `SparkListener` that writes stage, task, SQL plan and memory events as JSON lines. Jobs need
no code changes.

Build for Scala 2.12 and Spark 3.5:

```bash
sbt package
```

Attach it to a job:

```bash
spark-submit \
  --jars target/scala-2.12/tscm-listener_2.12-0.1.0.jar \
  --conf spark.extraListeners=org.apache.spark.tscm.TscmListener \
  --conf spark.tscm.output=/path/to/events \
  --conf spark.executor.processTreeMetrics.enabled=true \
  --conf spark.executor.metrics.pollingInterval=1s \
  --conf spark.sql.adaptive.enabled=false \
  job.py
```

Each application writes `<application id>.jsonl`. Load it with
`tscm.collect.store.read_events` and `ingest_listener`.
