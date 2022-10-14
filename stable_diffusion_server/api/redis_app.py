from stable_diffusion_server.api.base import AppConfig, create_app
from stable_diffusion_server.engine.repos.blob_repo import RedisBlobRepo
from stable_diffusion_server.engine.repos.key_value_repo import RedisKeyValueRepo
from stable_diffusion_server.engine.repos.messaging_repo import RedisMessagingRepo
from stable_diffusion_server.engine.repos.user_repo import RedisUserRepo

app_config = AppConfig(
    blob_repo_class=RedisBlobRepo,
    messaging_repo_class=RedisMessagingRepo,
    user_repo=RedisUserRepo,
    key_value_repo=RedisKeyValueRepo,
)

app = create_app(app_config)
