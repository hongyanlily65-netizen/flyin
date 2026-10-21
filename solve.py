from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
import sys


@dataclass
class Edge:
    target: str
    max_link_capacity: int = 1


@dataclass
class Zone:
    name: str
    x: int
    y: int
    kind: str = "normal"
    max_drones: int = 1
    connections: list[Edge] = field(default_factory=list)


@dataclass
class Drone:
    drone_id: int
    zone: str | None
    delivered: bool = False
    last_zone: str | None = None
    in_transit_from: str | None = None
    in_transit_to: str | None = None
    remaining_turns: int = 0


@dataclass
class MapData:
    nb_drones: int
    start: str
    goal: str
    zones: dict[str, Zone]


@dataclass
class Proposal:
    drone_id: int
    source: str
    target: str
    duration: int
    edge_capacity: int
    score: tuple[float, int, int, float, int, int, str]


def parse_map(path: str | Path) -> MapData:
    """Parse a Fly-in map file into a small graph structure."""

    map_path = Path(path)
    nb_drones: int | None = None
    start: str | None = None
    goal: str | None = None
    zones: dict[str, Zone] = {}
    connection_lines: list[tuple[int, str]] = []

    for line_number, raw_line in enumerate(map_path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("nb_drones:"):
            value = line.removeprefix("nb_drones:").strip()
            if not value.isdigit() or int(value) <= 0:
                raise ValueError(f"{map_path}:{line_number}: invalid nb_drones")
            nb_drones = int(value)
            continue

        if line.startswith("connection:"):
            connection_lines.append((line_number, line))
            continue

        prefix = None
        if line.startswith("start_hub:"):
            prefix = "start_hub:"
        elif line.startswith("end_hub:"):
            prefix = "end_hub:"
        elif line.startswith("hub:"):
            prefix = "hub:"

        if prefix is None:
            raise ValueError(f"{map_path}:{line_number}: invalid line: {line}")

        main, metadata = split_metadata(line.removeprefix(prefix).strip())
        parts = main.split()
        if len(parts) != 3:
            raise ValueError(f"{map_path}:{line_number}: invalid hub line")

        name, x_raw, y_raw = parts
        if "-" in name or " " in name:
            raise ValueError(f"{map_path}:{line_number}: invalid hub name")
        if name in zones:
            raise ValueError(f"{map_path}:{line_number}: duplicate hub {name}")

        meta = parse_metadata(metadata, line_number, map_path)
        kind = meta.get("zone", "normal")
        if kind not in {"normal", "blocked", "restricted", "priority"}:
            raise ValueError(f"{map_path}:{line_number}: invalid zone {kind}")

        max_drones = positive_int(meta.get("max_drones", "1"), "max_drones",
                                  line_number, map_path)
        zones[name] = Zone(
            name=name,
            x=int(x_raw),
            y=int(y_raw),
            kind=kind,
            max_drones=max_drones,
        )

        if prefix == "start_hub:":
            if start is not None:
                raise ValueError(f"{map_path}:{line_number}: duplicate start")
            start = name
        elif prefix == "end_hub:":
            if goal is not None:
                raise ValueError(f"{map_path}:{line_number}: duplicate goal")
            goal = name

    if nb_drones is None or start is None or goal is None:
        raise ValueError(f"{map_path}: missing nb_drones/start_hub/end_hub")

    seen_edges: set[tuple[str, str]] = set()
    for line_number, line in connection_lines:
        main, metadata = split_metadata(line.removeprefix("connection:").strip())
        if main.count("-") != 1:
            raise ValueError(f"{map_path}:{line_number}: invalid connection")
        left, right = main.split("-")
        if left not in zones or right not in zones:
            raise ValueError(f"{map_path}:{line_number}: unknown zone")

        edge_key = normalized_edge(left, right)
        if edge_key in seen_edges:
            raise ValueError(f"{map_path}:{line_number}: duplicate connection")
        seen_edges.add(edge_key)

        meta = parse_metadata(metadata, line_number, map_path)
        cap = positive_int(meta.get("max_link_capacity", "1"),
                           "max_link_capacity", line_number, map_path)
        zones[left].connections.append(Edge(target=right, max_link_capacity=cap))
        zones[right].connections.append(Edge(target=left, max_link_capacity=cap))

    zones[start].max_drones = max(zones[start].max_drones, nb_drones)
    zones[goal].max_drones = max(zones[goal].max_drones, nb_drones)
    return MapData(nb_drones=nb_drones, start=start, goal=goal, zones=zones)


def split_metadata(text: str) -> tuple[str, str | None]:
    if "[" not in text:
        return text, None
    main, rest = text.split("[", 1)
    if not rest.endswith("]"):
        raise ValueError("metadata must end with ]")
    return main.strip(), rest[:-1].strip()


def parse_metadata(
    text: str | None,
    line_number: int,
    map_path: Path,
) -> dict[str, str]:
    if text is None or text == "":
        return {}
    metadata: dict[str, str] = {}
    for item in text.split():
        if item.count("=") != 1:
            raise ValueError(f"{map_path}:{line_number}: invalid metadata")
        key, value = item.split("=")
        if key in metadata:
            raise ValueError(f"{map_path}:{line_number}: duplicate metadata")
        metadata[key] = value
    return metadata


def positive_int(value: str, key: str, line_number: int, map_path: Path) -> int:
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"{map_path}:{line_number}: {key} must be positive")
    return int(value)


def normalized_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def solve(map_data: MapData, max_turns: int | None = None) -> list[str]:
    """Return one output line per turn using greedy turn-by-turn simulation.

    The important idea is:
    - drones decide from nearest-to-goal to farthest-from-goal;
    - each drone scores all of its outgoing edges before choosing one;
    - reserved usage changes immediately, while positions change together.
    """

    distances = compute_goal_distances(map_data)
    if distances[map_data.start] == float("inf"):
        raise RuntimeError("goal is unreachable")

    drones = {
        drone_id: Drone(drone_id=drone_id, zone=map_data.start)
        for drone_id in range(1, map_data.nb_drones + 1)
    }
    output: list[str] = []
    turn_limit = max_turns or max(10, len(map_data.zones) * map_data.nb_drones * 8)

    for _turn in range(1, turn_limit + 1):
        if all(drone.delivered for drone in drones.values()):
            return output

        line, progress = simulate_one_turn(map_data, drones, distances)
        if line:
            output.append(line)
        if not progress:
            raise RuntimeError("solver stalled without progress")

    raise RuntimeError("solver exceeded maximum turn limit")


def simulate_one_turn(
    map_data: MapData,
    drones: dict[int, Drone],
    distances: dict[str, float],
) -> tuple[str, bool]:
    """Choose moves in front-to-back order, then apply them together."""

    reserved_zone_usage = count_zones(map_data, drones)
    reserved_edge_usage: defaultdict[tuple[str, str], int] = defaultdict(int)
    active_moves: list[tuple[int, str]] = []

    # Drones already on a restricted connection arrive at the end of this turn.
    # Their connection and destination capacity are unavailable to new plans.
    for drone in drones.values():
        if drone.in_transit_to is None:
            continue
        assert drone.in_transit_from is not None
        edge_key = normalized_edge(drone.in_transit_from, drone.in_transit_to)
        reserved_edge_usage[edge_key] += 1
        reserved_zone_usage[drone.in_transit_to] += 1
        active_moves.append(
            (drone.drone_id, f"{drone.in_transit_from}-{drone.in_transit_to}")
        )

    future_arrivals: defaultdict[str, int] = defaultdict(int)
    future_edges: defaultdict[tuple[str, str], int] = defaultdict(int)
    accepted: dict[int, Proposal] = {}

    decision_order = sorted(
        (
            drone for drone in drones.values()
            if not drone.delivered and drone.zone is not None
        ),
        key=lambda drone: (distances[drone.zone], drone.drone_id),
    )

    for drone in decision_order:
        proposals = build_drone_proposals(
            map_data, drone, distances, reserved_zone_usage
        )
        proposal = next(
            (
                candidate for candidate in proposals
                if proposal_is_legal(
                    map_data,
                    candidate,
                    reserved_zone_usage,
                    reserved_edge_usage,
                    future_arrivals,
                    future_edges,
                )
            ),
            None,
        )
        if proposal is None:
            continue

        accepted[proposal.drone_id] = proposal
        edge_key = normalized_edge(proposal.source, proposal.target)
        reserved_zone_usage[proposal.source] -= 1
        reserved_edge_usage[edge_key] += 1

        if proposal.duration == 1:
            reserved_zone_usage[proposal.target] += 1
            active_moves.append((proposal.drone_id, proposal.target))
        else:
            future_arrivals[proposal.target] += 1
            future_edges[edge_key] += 1
            active_moves.append(
                (proposal.drone_id, f"{proposal.source}-{proposal.target}")
            )

    for drone in drones.values():
        if drone.in_transit_to is not None:
            destination = drone.in_transit_to
            drone.zone = destination
            drone.last_zone = drone.in_transit_from
            drone.in_transit_from = None
            drone.in_transit_to = None
            drone.remaining_turns = 0
            drone.delivered = destination == map_data.goal
            continue

        proposal = accepted.get(drone.drone_id)
        if proposal is None or drone.delivered:
            continue

        drone.last_zone = proposal.source
        if proposal.duration == 1:
            drone.zone = proposal.target
            drone.delivered = proposal.target == map_data.goal
        else:
            drone.zone = None
            drone.in_transit_from = proposal.source
            drone.in_transit_to = proposal.target
            drone.remaining_turns = 1

    active_moves.sort(key=lambda item: item[0])
    line = " ".join(f"D{drone_id}-{destination}"
                    for drone_id, destination in active_moves)
    progress = bool(active_moves)
    return line, progress


def build_drone_proposals(
    map_data: MapData,
    drone: Drone,
    distances: dict[str, float],
    reserved_zone_usage: dict[str, int],
) -> list[Proposal]:
    """Return all reachable next edges for one drone, best score first."""

    proposals: list[Proposal] = []
    assert drone.zone is not None
    source = drone.zone
    source_distance = distances[source]

    for edge in map_data.zones[source].connections:
        target = edge.target
        target_zone = map_data.zones[target]
        target_distance = distances[target]
        if target_zone.kind == "blocked" or target_distance == float("inf"):
            continue

        progress_penalty = 0 if target_distance < source_distance else 1
        backtrack_penalty = 1 if target == drone.last_zone else 0
        capacity = zone_capacity(map_data, target)
        crowded = 0.0 if capacity == float("inf") else (
            reserved_zone_usage[target] / capacity
        )
        duration = move_duration(target_zone.kind)
        score = (
            target_distance,
            progress_penalty,
            zone_rank(target_zone.kind),
            crowded,
            backtrack_penalty,
            drone.drone_id,
            target,
        )
        proposals.append(Proposal(
            drone_id=drone.drone_id,
            source=source,
            target=target,
            duration=duration,
            edge_capacity=edge.max_link_capacity,
            score=score,
        ))

    proposals.sort(key=lambda proposal: proposal.score)
    return proposals


def proposal_is_legal(
    map_data: MapData,
    proposal: Proposal,
    reserved_zone_usage: dict[str, int],
    reserved_edge_usage: dict[tuple[str, str], int],
    future_arrivals: dict[str, int],
    future_edges: dict[tuple[str, str], int],
) -> bool:
    edge_key = normalized_edge(proposal.source, proposal.target)
    if reserved_edge_usage[edge_key] >= proposal.edge_capacity:
        return False

    if proposal.duration == 1:
        return (reserved_zone_usage[proposal.target] + 1
                <= zone_capacity(map_data, proposal.target))

    if (reserved_zone_usage[proposal.target]
            + future_arrivals[proposal.target] + 1
            > zone_capacity(map_data, proposal.target)):
        return False
    if future_edges[edge_key] + 1 > proposal.edge_capacity:
        return False
    return True


def compute_goal_distances(map_data: MapData) -> dict[str, float]:
    distances = {name: float("inf") for name in map_data.zones}
    distances[map_data.goal] = 0.0
    queue: list[tuple[float, str]] = [(0.0, map_data.goal)]

    while queue:
        current_distance, current = heappop(queue)
        if current_distance > distances[current]:
            continue
        if map_data.zones[current].kind == "blocked":
            continue

        for edge in map_data.zones[current].connections:
            neighbor = edge.target
            if map_data.zones[neighbor].kind == "blocked":
                continue
            candidate = current_distance + move_duration(map_data.zones[current].kind)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heappush(queue, (candidate, neighbor))

    return distances


def count_zones(map_data: MapData, drones: dict[int, Drone]) -> dict[str, int]:
    counts = {name: 0 for name in map_data.zones}
    for drone in drones.values():
        if not drone.delivered and drone.zone is not None:
            counts[drone.zone] += 1
    return counts


def zone_capacity(map_data: MapData, zone_name: str) -> float:
    if zone_name in {map_data.start, map_data.goal}:
        return float("inf")
    return float(map_data.zones[zone_name].max_drones)


def move_duration(zone_kind: str) -> int:
    return 2 if zone_kind == "restricted" else 1


def zone_rank(zone_kind: str) -> int:
    ranks = {"priority": 0, "normal": 1, "restricted": 2, "blocked": 3}
    return ranks[zone_kind]


def main() -> None:
    map_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("maps/easy/01_linear_path.txt")
    )
    map_data = parse_map(map_path)
    for line in solve(map_data):
        print(line)


if __name__ == "__main__":
    main()
