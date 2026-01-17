import asyncio
from avatar_service.server import _load_config
from redis import Redis, ResponseError
from loguru import logger
from temporalio.client import Client as TemporalClient
from avatar_service.application.workflows.CreateAvatarWf import CreateAvatarWf


class WorkflowDispatcher:
    def __init__(self):
        self._configuration_params = _load_config()
        self._redis_client =  Redis(
            host=self._configuration_params.REDIS_PARAMETERS.HOST,
            port=self._configuration_params.REDIS_PARAMETERS.PORT,
            decode_responses=True
        )
        self._temporal_client = None

    def _create_consumer_group(self):
        try:
            self._redis_client.xgroup_create(
                self._configuration_params.REDIS_PARAMETERS.STREAM_NAME,
                self._configuration_params.REDIS_PARAMETERS.CONSUMER_GROUP_NAME,
                id="$",
                mkstream=True
            )
            logger.info("Group created")
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.warning("Group exists")
            else:
                raise

    def _get_consumer_instances(self):
        group = list(self._redis_client.xinfo_groups(self._configuration_params.REDIS_PARAMETERS.STREAM_NAME))[0]
        return int(group["consumers"])

    async def _create_temporal_client(self):
        return await TemporalClient.connect(
            self._configuration_params.TEMPORAL_PARAMETERS.SERVER_URL
        )

    async def run(self):
        self._temporal_client = await self._create_temporal_client()
        self._create_consumer_group()
        active_consumer_instances = self._get_consumer_instances()
        if active_consumer_instances >= self._configuration_params.TEMPORAL_PARAMETERS.MAX_WORKERS:
            return

        consumer_name = f"avatar_worker_{active_consumer_instances + 1}"
        while True:
            messages = self._redis_client.xreadgroup(
                self._configuration_params.REDIS_PARAMETERS.CONSUMER_GROUP_NAME,
                consumer_name,
                streams={self._configuration_params.REDIS_PARAMETERS.STREAM_NAME: ">"},
                count=1,
                block=self._configuration_params.REDIS_PARAMETERS.BLOCK_MILLISECONDS,
            )
            if not messages:
                continue

            for stream, entries in messages:
                for entry_id, fields in entries:
                    logger.info(f"{consumer_name} processing {entry_id}: {fields}")
                    await self._temporal_client.start_workflow(
                        workflow=CreateAvatarWf.run,
                        args=[fields],
                        id=entry_id,
                        task_queue=self._configuration_params.TEMPORAL_PARAMETERS.TASK_QUEUE,
                    )
                    self._redis_client.xack(
                        self._configuration_params.REDIS_PARAMETERS.STREAM_NAME,
                        self._configuration_params.REDIS_PARAMETERS.CONSUMER_GROUP_NAME,
                        entry_id,
                    )

if __name__ == "__main__":
    dispatcher = WorkflowDispatcher()
    asyncio.run(dispatcher.run())