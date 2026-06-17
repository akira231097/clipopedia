# Deployment

Clip'O'pedia is a single long-running worker (it polls a queue), which makes it
a natural fit for a container scheduler such as **AWS ECS Fargate**. The bot
holds no state of its own — all state lives in the queue, the vector store, and
the metadata DB — so it scales horizontally and restarts cleanly.

## Container

```bash
docker build -t clipopedia:latest .
# Push to your registry (example: ECR)
docker tag clipopedia:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/clipopedia:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/clipopedia:latest
```

## ECS task definition

[`ecs-task-definition.json`](ecs-task-definition.json) is a **template**. Every
sensitive value is a `<PLACEHOLDER>`:

- Non-secret config is passed as plain `environment` entries.
- Secrets are injected at runtime from **AWS Secrets Manager** via `valueFrom`
  ARNs — they are never baked into the image or committed to source control.

Register it and roll out:

```bash
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json
aws ecs update-service --cluster <CLUSTER> --service clipopedia --force-new-deployment
```

## Configuration

All runtime configuration is environment-driven — see [`.env.example`](../.env.example)
for the full list and [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for what
each component talks to.
