tests-up:
	docker-compose up --force-recreate --build api redis_worker e2e_tests

tests-run:
	docker-compose run e2e_tests
