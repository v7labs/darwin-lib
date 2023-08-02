import os
from pathlib import Path
from venv import create

from dotenv import load_dotenv

from e2e_tests import teardown_tests
from e2e_tests.conftest import ConfigValues
from e2e_tests.setup_tests import (
    create_dataset,
    create_item,
    create_random_image,
    teardown_tests,
)

# TODO List:
# 1. Dogfood creating dataset
# 2. Dogfood creating item
# 3. Dogfood creating annotation


def main() -> None:
    path = Path(__file__).parent / "e2e_tests" / ".env"
    assert path.exists(), f"Path {path} does not exist"

    load_dotenv(path)

    host, key, team = os.getenv("E2E_ENVIRONMENT"), os.getenv("E2E_API_KEY"), os.getenv("E2E_TEAM_SLUG")
    config_values = ConfigValues(api_key=key, server=host, team_slug=team)

    dataset = create_dataset("test", config_values)

    image = create_random_image("test", Path(__file__).parent)

    # item = create_item(dataset.slug, "test", image, config_values)

    print(dataset)
    # print(item)

    teardown_tests(config_values, [dataset])


if __name__ == "__main__":
    main()
