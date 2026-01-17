import asyncio
from avatar_service.application.workflows.CreateAvatarWf import CreateAvatarWf
from avatar_service.application.activities.image_processing_activity import save_user_image
from avatar_service.application.activities.econ_activity import run_econ_inference
from avatar_service.application.activities.texture_generation_activity import generate_realistic_textures
from avatar_service.application.activities.storage_activity import upload_avatar_to_s3
from avatar_service.server import _load_config
from loguru import logger
from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker as TemporalWorker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

configuration_params = _load_config()

async def _connect_to_temporal_server():
    return await TemporalClient.connect(configuration_params.TEMPORAL_PARAMETERS.SERVER_URL)

async def start_worker():
    restrictions = SandboxRestrictions.default.with_passthrough_modules()
    temporal_client = await _connect_to_temporal_server()
    temporal_worker = TemporalWorker(
        temporal_client,
        task_queue=configuration_params.TEMPORAL_PARAMETERS.TASK_QUEUE,
        workflows=[CreateAvatarWf],
        activities=[
            save_user_image,
            run_econ_inference,
            generate_realistic_textures,
            upload_avatar_to_s3,
        ],
        workflow_runner=SandboxedWorkflowRunner(restrictions=restrictions),
    )
    await temporal_worker.run()

if __name__ == "__main__":
    asyncio.run(start_worker())