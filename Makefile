test:
	docker-compose --env-file .env.test up --abort-on-container-exit --exit-code-from e2e_tests --force-recreate --build api redis_worker e2e_tests

run:
	docker-compose up --abort-on-container-exit --force-recreate --build api redis_worker
