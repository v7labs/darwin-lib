from typing import List, Optional, Tuple

from darwin.future.core.client import ClientCore
from darwin.future.data_objects.team import TeamCore, TeamMemberCore


def get_team(client: ClientCore, team_slug: Optional[str] = None) -> TeamCore:
    """
    Returns a TeamCore object for the specified team slug.

    Parameters:
        client (ClientCore): The client to use for the request.
        team_slug (Optional[str]): The slug of the team to get. If not specified, the
            default team from the client's config will be used.

    Returns:
        TeamCore: The TeamCore object for the specified team slug.

    Raises:
        HTTPError: If the response status code is not in the 200-299 range.
    """
    if not team_slug:
        team_slug = client.config.default_team
    response = client.get(f"/teams/{team_slug}/")
    return TeamCore.parse_obj(response)


def get_team_members(
    client: ClientCore,
) -> Tuple[List[TeamMemberCore], List[Exception]]:
    """
    Returns a tuple containing a list of TeamMemberCore objects and a list of exceptions
    that occurred while parsing the response.

    Parameters:
        client (ClientCore): The client to use for the request.

    Returns:
        Tuple[List[TeamMemberCore], List[Exception]]: A tuple containing a list of
            TeamMemberCore objects and a list of exceptions that occurred while parsing
            the response.

    Raises:
        HTTPError: If the response status code is not in the 200-299 range.
    """
    response = client.get("/memberships")
    members = []
    errors = []
    for item in response:
        try:
            members.append(TeamMemberCore.parse_obj(item))
        except Exception as e:
            errors.append(e)
    return (members, errors)
