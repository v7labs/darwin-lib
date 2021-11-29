import logging
import os
import time
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, cast
from urllib import parse

import requests
from requests import Response

from darwin.config import Config
from darwin.dataset import RemoteDataset
from darwin.dataset.identifier import DatasetIdentifier
from darwin.datatypes import DarwinVersionNumber, Feature, Team
from darwin.exceptions import (
    InsufficientStorage,
    InvalidLogin,
    MissingConfig,
    NameTaken,
    NotFound,
    Unauthorized,
    ValidationError,
)
from darwin.utils import is_project_dir, urljoin


class Client:
    def __init__(self, config: Config, log: Logger, default_team: Optional[str] = None):
        self.config: Config = config
        self.url: str = config.get("global/api_endpoint")
        self.base_url: str = config.get("global/base_url")
        self.default_team: str = default_team or config.get("global/default_team")
        self.features: Dict[str, List[Feature]] = {}
        self._newer_version: Optional[DarwinVersionNumber] = None
        self.log = log

    def _get_raw(self, endpoint: str, team: Optional[str] = None, retry: bool = False) -> Response:
        response: Response = requests.get(urljoin(self.url, endpoint), headers=self._get_headers(team))

        self.log.debug(
            f"Client GET request response ({response.text}) with status "
            f"({response.status_code}). "
            f"Client: ({self})"
            f"Request: (endpoint={endpoint})"
        )

        self._raise_if_known_error(response, endpoint)

        if not response.ok and retry:
            time.sleep(10)
            return self._get_raw(endpoint=endpoint, retry=False)

        response.raise_for_status()

        return response

    def _get(
        self, endpoint: str, team: Optional[str] = None, retry: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get something from the server through HTTP

        Parameters
        ----------
        endpoint : str
            Recipient of the HTTP operation
        retry : bool
            Retry to perform the operation. Set to False on recursive calls.


        Returns
        -------
        Union[Dict[str, Any], List[Dict[str, Any]]]
            Dictionary which contains the server response

        Raises
        ------
        NotFound
            Resource not found
        Unauthorized
            Action is not authorized
        """

        response = self._get_raw(endpoint, team, retry)
        return self._decode_response(response)

    def _put_raw(
        self, endpoint: str, payload: Dict[str, Any], team: Optional[str] = None, retry: bool = False
    ) -> Response:
        response: requests.Response = requests.put(
            urljoin(self.url, endpoint), json=payload, headers=self._get_headers(team)
        )

        self.log.debug(
            f"Client PUT request got response ({response.text}) with status "
            f"({response.status_code}). "
            f"Client: ({self})"
            f"Request: (endpoint={endpoint}, payload={payload})"
        )

        self._raise_if_known_error(response, endpoint)

        if not response.ok and retry:
            time.sleep(10)
            return self._put_raw(endpoint, payload=payload, retry=False)

        response.raise_for_status()

        return response

    def _put(
        self, endpoint: str, payload: Dict[str, Any], team: Optional[str] = None, retry: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Put something on the server trough HTTP

        Parameters
        ----------
        endpoint : str
            Recipient of the HTTP operation
        payload : dict
            What you want to put on the server (typically json encoded)
        retry : bool
            Retry to perform the operation. Set to False on recursive calls.

        Returns
        -------
        dict
            Dictionary which contains the server response
        """
        response = self._put_raw(endpoint, payload, team, retry)
        return self._decode_response(response)

    def _post(
        self, endpoint: str, payload: Optional[Dict[Any, Any]] = None, team: Optional[str] = None, retry: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Post something new on the server trough HTTP

        Parameters
        ----------
        endpoint : str
            Recipient of the HTTP operation
        payload : dict
            What you want to put on the server (typically json encoded)
        retry : bool
            Retry to perform the operation. Set to False on recursive calls.
        refresh : boolself._raise_if_known_error(response, endpoint)
        Returns
        -------
        dict
        Dictionary which contains the server response
        """
        if payload is None:
            payload = {}

        response: Response = requests.post(urljoin(self.url, endpoint), json=payload, headers=self._get_headers(team))

        self.log.debug(
            f"Client POST request response ({response.json()}) with unexpected status "
            f"({response.status_code}). "
            f"Client: ({self})"
            f"Request: (endpoint={endpoint}, payload={payload})"
        )

        self._raise_if_known_error(response, endpoint)

        if not response.ok and retry:
            time.sleep(10)
            return self._post(endpoint, payload=payload, retry=False)

        response.raise_for_status()

        return self._decode_response(response)

    def _delete(
        self, endpoint: str, payload: Optional[Dict[Any, Any]] = None, team: Optional[str] = None, retry: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Delete something new on the server trough HTTP

        Parameters
        ----------
        endpoint : str
            Recipient of the HTTP operation.
        payload : Optional[Dict[Any, Any]]
            JSON-encoded extra information that might be necessary to perform the deletion.
        team : Optional[str]
            Optional team slug, used to build the request headers.
        retry : bool
            Retry to perform the operation. Set to False on recursive calls.

        Returns
        -------
        dict
        Dictionary which contains the server response
        """
        if payload is None:
            payload = {}

        response: requests.Response = requests.delete(
            urljoin(self.url, endpoint), json=payload, headers=self._get_headers(team)
        )

        self.log.debug(
            f"Client DELETE request response ({response.json()}) with unexpected status "
            f"({response.status_code}). "
            f"Client: ({self})"
            f"Request: (endpoint={endpoint})"
        )

        self._raise_if_known_error(response, endpoint)

        if not response.ok and retry:
            time.sleep(10)
            return self._delete(endpoint, payload=payload, retry=False)

        response.raise_for_status()

        return self._decode_response(response)

    def _raise_if_known_error(self, response: Response, endpoint: str) -> None:

        if response.status_code == 401:
            raise Unauthorized()

        if response.status_code == 404:
            raise NotFound(urljoin(self.url, endpoint))

        is_json = response.headers.get("content-type") == "application/json"
        if is_json:
            body = response.json()
            is_name_taken: Optional[bool] = None
            if isinstance(body, Dict):
                is_name_taken = body.get("errors", {}).get("name") == ["has already been taken"]

            if response.status_code == 422:
                if is_name_taken:
                    raise NameTaken
                raise ValidationError(body)

        if response.status_code == 429:
            error_code: Optional[str] = None
            try:
                error_code = response.json()["errors"]["code"]
            except:
                pass

            if error_code == "INSUFFICIENT_REMAINING_STORAGE":
                raise InsufficientStorage()

    def list_local_datasets(self, team: Optional[str] = None) -> Iterator[Path]:
        """
        Returns a list of all local folders which are detected as dataset.

        Returns
        -------
        list[Path]
        List of all local datasets
        """

        team_configs: List[Team] = []
        if team:
            team_data: Optional[Team] = self.config.get_team(team)
            if team_data:
                team_configs.append(team_data)
        else:
            team_configs = self.config.get_all_teams()

        for team_config in team_configs:
            projects_team: Path = Path(team_config.datasets_dir) / team_config.slug
            for project_path in projects_team.glob("*"):
                if project_path.is_dir() and is_project_dir(project_path):
                    yield Path(project_path)

    def list_remote_datasets(self, team: Optional[str] = None) -> Iterator[RemoteDataset]:
        """
        Returns a list of all available datasets with the team currently authenticated against.

        Returns
        -------
        list[RemoteDataset]
        List of all remote datasets
        """
        response: List[Dict[str, Any]] = cast(List[Dict[str, Any]], self._get("/datasets/", team=team))

        for dataset in response:
            yield RemoteDataset(
                name=dataset["name"],
                slug=dataset["slug"],
                team=team or self.default_team,
                dataset_id=dataset["id"],
                item_count=dataset["num_images"] + dataset["num_videos"],
                progress=dataset["progress"],
                client=self,
            )

    def get_remote_dataset(self, dataset_identifier: Union[str, DatasetIdentifier]) -> RemoteDataset:
        """
        Get a remote dataset based on the parameter passed.

        Parameters
        ----------
        dataset_identifier : Union[str, DatasetIdentifier]
            Identifier of the dataset. Can be the string version or a DatasetIdentifier object.

        Returns
        -------
        RemoteDataset
            Initialized dataset
        """
        parsed_dataset_identifier: DatasetIdentifier = DatasetIdentifier.parse(dataset_identifier)

        if not parsed_dataset_identifier.team_slug:
            parsed_dataset_identifier.team_slug = self.default_team

        try:
            matching_datasets: List[RemoteDataset] = [
                dataset
                for dataset in self.list_remote_datasets(team=parsed_dataset_identifier.team_slug)
                if dataset.slug == parsed_dataset_identifier.dataset_slug
            ]
        except Unauthorized:
            # There is a chance that we tried to access an open dataset
            dataset: Dict[str, Any] = cast(
                Dict[str, Any],
                self._get(f"{parsed_dataset_identifier.team_slug}/{parsed_dataset_identifier.dataset_slug}"),
            )

            # If there isn't a record of this team, create one.
            if not self.config.get_team(parsed_dataset_identifier.team_slug, raise_on_invalid_team=False):
                datasets_dir: Path = Path.home() / ".darwin" / "datasets"
                self.config.set_team(
                    team=parsed_dataset_identifier.team_slug, api_key="", datasets_dir=str(datasets_dir)
                )

            return RemoteDataset(
                name=dataset["name"],
                slug=dataset["slug"],
                team=parsed_dataset_identifier.team_slug,
                dataset_id=dataset["id"],
                item_count=dataset["num_images"] + dataset["num_videos"],
                progress=0,
                client=self,
            )
        if not matching_datasets:
            raise NotFound(parsed_dataset_identifier)
        return matching_datasets[0]

    def create_dataset(self, name: str, team: Optional[str] = None) -> RemoteDataset:
        """Create a remote dataset

        Parameters
        ----------
        name : str
            Name of the dataset to create

        Returns
        -------
        RemoteDataset
        The created dataset
        """
        dataset: Dict[str, Any] = cast(Dict[str, Any], self._post("/datasets", {"name": name}, team=team))
        return RemoteDataset(
            name=dataset["name"],
            team=team or self.default_team,
            slug=dataset["slug"],
            dataset_id=dataset["id"],
            item_count=dataset["num_images"],
            progress=0,
            client=self,
        )

    def archive_remote_dataset(self, dataset_id: int, team_slug: str) -> None:
        self._put(f"datasets/{dataset_id}/archive", payload={}, team=team_slug)

    def fetch_remote_files(
        self, dataset_id: int, cursor: Dict[str, Any], payload: Any, team_slug: str
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = cast(
            Dict[str, Any], self._post(f"/datasets/{dataset_id}/items?{parse.urlencode(cursor)}", payload, team_slug)
        )
        return response

    def fetch_remote_classes(self, team: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches all remote classes on the remote dataset"""
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        team_slug: str = the_team.slug
        response: Dict[str, Any] = cast(
            Dict[str, Any], self._get(f"/teams/{team_slug}/annotation_classes?include_tags=true")
        )

        return response["annotation_classes"]

    def update_annotation_class(self, class_id: int, payload: Any) -> Dict[str, Any]:
        response: Dict[str, Any] = cast(Dict[str, Any], self._put(f"/annotation_classes/{class_id}", payload))
        return response

    def create_annotation_class(self, dataset_id: int, type_ids: List[int], name: str) -> Dict[str, Any]:
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._post(
                "/annotation_classes",
                payload={
                    "dataset_id": dataset_id,
                    "name": name,
                    "metadata": {"_color": "auto"},
                    "annotation_type_ids": type_ids,
                    "datasets": [{"id": dataset_id}],
                },
            ),
        )
        return response

    def fetch_remote_attributes(self, dataset_id: int) -> List[Dict[str, Any]]:
        response: List[Dict[str, Any]] = cast(List[Dict[str, Any]], self._get(f"/datasets/{dataset_id}/attributes"))
        return response

    def load_feature_flags(self, team: Optional[str] = None) -> None:
        """Gets current features enabled for a team"""
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        team_slug: str = the_team.slug
        self.features[team_slug] = self.get_team_features(team_slug)

    def get_team_features(self, team_slug: str) -> List[Feature]:
        """
        Gets all the features for the given team together with their statuses.

        Parameters
        ----------
        team_slug : str
            Slug of the team.

        Returns
        -------
        List[FeaturePayload]
            List of feature for the given team.
        """
        response: List[Dict[str, Any]] = cast(List[Dict[str, Any]], self._get(f"/teams/{team_slug}/features"))

        features: List[Feature] = []
        for feature in response:
            features.append(Feature(name=str(feature["name"]), enabled=bool(feature["enabled"])))

        return features

    def feature_enabled(self, feature_name: str, team: Optional[str] = None) -> bool:
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return False

        team_slug: str = the_team.slug

        if team_slug not in self.features:
            self.load_feature_flags(team)

        team_features: List[Feature] = self.features[team_slug]
        for feature in team_features:
            if feature.name == feature_name:
                return feature.enabled

        return False

    def get_datasets_dir(self, team: Optional[str] = None) -> Optional[str]:
        """Gets the dataset directory of the specified team or the default one

        Parameters
        ----------
        team: str
            Team to get the directory from

        Returns
        -------
        str
            Path of the datasets for the selected team or the default one
        """
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        return the_team.datasets_dir

    def set_datasets_dir(self, datasets_dir: Path, team: Optional[str] = None) -> None:
        """Sets the dataset directory of the specified team or the default one

        Parameters
        ----------
        datasets_dir: Path
            Path to set as dataset directory of the team
        team: str
            Team to change the directory to
        """
        self.config.put(f"teams/{team or self.default_team}/datasets_dir", datasets_dir)

    def confirm_upload(self, dataset_item_id: int, team: Optional[str] = None) -> None:
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        team_slug: str = the_team.slug

        self._put_raw(endpoint=f"/dataset_items/{dataset_item_id}/confirm_upload", payload={}, team=team_slug)

    def sign_upload(self, dataset_item_id: int, team: Optional[str] = None) -> Optional[Dict[str, Any]]:
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        team_slug: str = the_team.slug

        response: Dict[str, Any] = cast(
            Dict[str, Any], self._get(f"/dataset_items/{dataset_item_id}/sign_upload", team=team_slug)
        )
        return response

    def upload_data(self, dataset_slug: str, payload: Any, team: Optional[str] = None) -> Optional[Dict[str, Any]]:
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        team_slug: str = the_team.slug

        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._put(endpoint=f"/teams/{team_slug}/datasets/{dataset_slug}/data", payload=payload, team=team_slug),
        )
        return response

    def annotation_types(self) -> List[Dict[str, Any]]:
        response: List[Dict[str, Any]] = cast(List[Dict[str, Any]], self._get("/annotation_types"))
        return response

    def get_exports(self, dataset_id: int, team: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        team_slug: str = the_team.slug

        response: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], self._get(f"/datasets/{dataset_id}/exports", team=team_slug)
        )
        return response

    def create_export(self, dataset_id: int, payload: Any, team_slug: str) -> None:
        self._post(f"/datasets/{dataset_id}/exports", payload=payload, team=team_slug)

    def get_report(self, dataset_id: int, granularity: str, team: Optional[str] = None) -> Optional[Response]:
        the_team: Optional[Team] = self.config.get_team(team or self.default_team)

        if not the_team:
            return None

        team_slug: str = the_team.slug

        return self._get_raw(
            f"/reports/{team_slug}/annotation?group_by=dataset,user&dataset_ids={dataset_id}&granularity={granularity}&format=csv&include=dataset.name,user.first_name,user.last_name,user.email",
            team_slug,
        )

    def delete_item(self, dataset_slug: str, team_slug: str, payload: Any) -> None:
        self._delete(f"teams/{team_slug}/datasets/{dataset_slug}/items", payload, team_slug)

    def archive_item(self, dataset_slug: str, team_slug: str, payload: Any) -> None:
        self._put(f"teams/{team_slug}/datasets/{dataset_slug}/items/archive", payload, team_slug)

    def restore_archived_item(self, dataset_slug: str, team_slug: str, payload: Any) -> None:
        self._put(f"teams/{team_slug}/datasets/{dataset_slug}/items/restore", payload, team_slug)

    def move_item_to_new(self, dataset_slug: str, team_slug: str, payload: Any) -> None:
        self._put(f"teams/{team_slug}/datasets/{dataset_slug}/items/move_to_new", payload, team_slug)

    def reset_item(self, dataset_slug: str, team_slug: str, payload: Any) -> None:
        self._put(f"teams/{team_slug}/datasets/{dataset_slug}/items/reset", payload, team_slug)

    def _get_headers(self, team: Optional[str] = None) -> Dict[str, str]:
        """Get the headers of the API calls to the backend.

        Parameters
        ----------

        Returns
        -------
        dict
        Contains the Content-Type and Authorization token
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        api_key: Optional[str] = None
        team_config: Optional[Team] = self.config.get_team(team or self.default_team, raise_on_invalid_team=False)

        if team_config:
            api_key = team_config.api_key

        if api_key and len(api_key) > 0:
            headers["Authorization"] = f"ApiKey {api_key}"

        from darwin import __version__

        headers["User-Agent"] = f"darwin-py/{__version__}"
        return headers

    @classmethod
    def local(cls, team_slug: Optional[str] = None) -> "Client":
        """
        Factory method to use the default configuration file to init the client

        Returns
        -------
        Client
        The initialized client
        """
        config_path: Path = Path.home() / ".darwin" / "config.yaml"
        return Client.from_config(config_path, team_slug=team_slug)

    @classmethod
    def from_config(cls, config_path: Path, team_slug: Optional[str] = None) -> "Client":
        """
        Factory method to create a client from the configuration file passed as parameter

        Parameters
        ----------
        config_path : str
            Path to a configuration file to use to create the client

        Returns
        -------
        Client
        The initialized client
        """
        if not config_path.exists():
            raise MissingConfig()
        config = Config(config_path)
        log = logging.getLogger()

        return cls(config=config, log=log, default_team=team_slug)

    @classmethod
    def from_guest(cls, datasets_dir: Optional[Path] = None) -> "Client":
        """
        Factory method to create a client and access datasets as a guest

        Parameters
        ----------
        datasets_dir : str
            String where the client should be initialized from (aka the root path)

        Returns
        -------
        Client
            The initialized client
        """
        if datasets_dir is None:
            datasets_dir = Path.home() / ".darwin" / "datasets"

        config: Config = Config(path=None)
        config.set_global(api_endpoint=Client.default_api_url(), base_url=Client.default_base_url())
        log = logging.getLogger()

        return cls(config=config, log=log)

    @classmethod
    def from_api_key(cls, api_key: str, datasets_dir: Optional[Path] = None) -> "Client":
        """
        Factory method to create a client given an API key

        Parameters
        ----------
        api_key: str
            API key to use to authenticate the client
        datasets_dir : str
            String where the client should be initialized from (aka the root path)

        Returns
        -------
        Client
            The initialized client
        """
        if not datasets_dir:
            datasets_dir = Path.home() / ".darwin" / "datasets"

        headers: Dict[str, str] = {"Content-Type": "application/json", "Authorization": f"ApiKey {api_key}"}
        api_url: str = Client.default_api_url()
        response: requests.Response = requests.get(urljoin(api_url, "/users/token_info"), headers=headers)

        if not response.ok:
            raise InvalidLogin()

        data: Dict[str, Any] = response.json()
        team: str = data["selected_team"]["slug"]

        config: Config = Config(path=None)
        config.set_team(team=team, api_key=api_key, datasets_dir=str(datasets_dir))
        config.set_global(api_endpoint=api_url, base_url=Client.default_base_url())
        log = logging.getLogger()

        return cls(config=config, log=log, default_team=team)

    @staticmethod
    def default_api_url() -> str:
        """Returns the default api url"""
        return f"{Client.default_base_url()}/api/"

    @staticmethod
    def default_base_url() -> str:
        """Returns the default base url"""
        return os.getenv("DARWIN_BASE_URL", "https://darwin.v7labs.com")

    def _decode_response(self, response: requests.Response) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Decode the response as JSON entry or return a dictionary with the error

        Parameters
        ----------
        response: requests.Response
            Response to decode
        debug : bool
            Debugging flag. In this case failed requests get printed

        Returns
        -------
        dict
        JSON decoded entry or error
        """

        if "latest-darwin-py" in response.headers:
            self._handle_latest_darwin_py(response.headers["latest-darwin-py"])

        try:
            return response.json()
        except ValueError:
            self.log.error(f"[ERROR {response.status_code}] {response.text}")
            response.close()
            return {"error": "Response is not JSON encoded", "status_code": response.status_code, "text": response.text}

    def _handle_latest_darwin_py(self, server_latest_version: str) -> None:
        try:

            def parse_version(version: str) -> DarwinVersionNumber:
                (major, minor, patch) = version.split(".")
                return (int(major), int(minor), int(patch))

            from darwin import __version__

            current_version = parse_version(__version__)
            latest_version = parse_version(server_latest_version)
            if current_version >= latest_version:
                return
            self._newer_version = latest_version
        except:
            pass

    @property
    def newer_darwin_version(self) -> Optional[DarwinVersionNumber]:
        return self._newer_version

    def __str__(self) -> str:
        return f"Client(default_team={self.default_team})"
