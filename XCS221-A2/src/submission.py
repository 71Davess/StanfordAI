from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch


# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Check out the docstring for `State` in `util.py` for more details and code.

########################################################################################
# Problem 2a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # ### START CODE HERE ###
        # The search begins at the provided starting location; no extra memory needed.
        return State(location=self.startLocation, memory=None)
        # ### END CODE HERE ###

    def isEnd(self, state: State) -> bool:
        # ### START CODE HERE ###
        # End state reached when the current location contains the target tag.
        return self.endTag in self.cityMap.tags[state.location]
        # ### END CODE HERE ###

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # ### START CODE HERE ###
        # One successor per adjacent location, carrying over no extra memory.
        successors = []
        for neighbour, distance in self.cityMap.distances[state.location].items():
            successors.append(
                (neighbour, State(location=neighbour, memory=None), float(distance))
            )
        return successors
        # ### END CODE HERE ###


########################################################################################
# Problem 2b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. If you prefer, you may create a new map using via
    `createCustomMap()`.

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "park", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # Or, if you would rather use a custom map, you can uncomment the following!
    # cityMap = createCustomMap("data/custom.pbf", "data/custom-landmarks".json")

    startLocation, endTag = None, None

    # ### START CODE HERE ###
    # Route from Gates building to the Stanford Stadium.
    startLocation = locationFromTag(makeTag("landmark", "gates"), cityMap)
    endTag = makeTag("landmark", "stanford_stadium")
    # ### END CODE HERE ###
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 3a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        pass
        # ### START CODE HERE ###
        # Track which waypoint tags still need to be visited and on the plan. Remove from
        # remaining any waypoint tags already covered at the start location.
        remaining = set(self.waypointTags)
        remaining -= set(self.cityMap.tags[self.startLocation])
        return State(location=self.startLocation, memory=tuple(sorted(remaining)))
        # ### END CODE HERE ###

    def isEnd(self, state: State) -> bool:
        pass
        # ### START CODE HERE ###
        # End at the location with the end tag and all waypoints covered.
        return (
            self.endTag in self.cityMap.tags[state.location]
            and len(state.memory) == 0
        )
        # ### END CODE HERE ###

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        pass
        # ### START CODE HERE ###
        successors = []
        for neighbour, distance in self.cityMap.distances[state.location].items():
            remaining = set(state.memory)
            remaining -= set(self.cityMap.tags[neighbour])
            newState = State(location=neighbour, memory=tuple(sorted(remaining)))
            successors.append((neighbour, newState, float(distance)))
        return successors
        # ### END CODE HERE ###


########################################################################################
# Problem 3c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 2b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()

    startTag = None
    startLocation = None
    waypointTags = None
    endTag = None

    # ### START CODE HERE ###
    # Start at Gates, visit several places on the way to the Oval (end point).
    startTag = makeTag("landmark", "gates")
    startLocation = locationFromTag(startTag, cityMap)

    waypointTags = [
        makeTag("landmark", "hoover_tower"),
        makeTag("landmark", "tressider"),
        makeTag("label", "1758004045"),
        makeTag("landmark", "memorial_church"),
    ]

    endTag = makeTag("landmark", "oval")
    # ### END CODE HERE ###
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 4a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            pass
            # ### START CODE HERE ###
            return problem.startState()
            # ### END CODE HERE ###

        def isEnd(self, state: State) -> bool:
            pass
            # ### START CODE HERE ###
            return problem.isEnd(state)
            # ### END CODE HERE ###

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            pass
            # ### START CODE HERE ###
            adjusted_succ = []
            currentH = heuristic.evaluate(state)
            for action, newState, cost in problem.successorsAndCosts(state):
                # Edge cost adjusted by heuristic difference to simulate A* within UCS.
                adjustedCost = cost + heuristic.evaluate(newState) - currentH
                adjusted_succ.append((action, newState, adjustedCost))
            return adjusted_succ
            # ### END CODE HERE ###

    return NewSearchProblem()


########################################################################################
# Problem 4b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # ### START CODE HERE ###
        # Cache geolocations of all end-tag locations for fast min-distance lookup.
        self.endLocations = [
            loc for loc, tags in self.cityMap.tags.items() if self.endTag in tags
        ]
        # ### END CODE HERE ###

    def evaluate(self, state: State) -> float:
        pass
        # ### START CODE HERE ###
        # Return straight-line distance to the closest end-tag location.
        here = self.cityMap.geoLocations[state.location]
        return min(
            computeDistance(here, self.cityMap.geoLocations[loc])
            for loc in self.endLocations
        )
        # ### END CODE HERE ###


########################################################################################
# Problem 4c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `startLocation`.
        class ReverseShortestPathProblem(SearchProblem):
            def startState(self) -> State:
                """
                Return special "END" state
                """
                pass
                # ### START CODE HERE ###
                return State(location="END", memory=None)
                # ### END CODE HERE ###

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False), 
                UCS will exhaustively compute costs to *all* other states.
                """
                pass
                # ### START CODE HERE ###
                return False
                # ### END CODE HERE ###

            def successorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                # If current location is the special "END" state, 
                # return all the locations with the desired endTag and cost 0 
                # (i.e, we connect the special location "END" with cost 0 to all locations with endTag)
                # Else, return all the successors of current location and their corresponding distances according to the cityMap
                pass
                # ### START CODE HERE ###
                if state.location == "END":
                    return [
                        (loc, State(location=loc, memory=None), 0.0)
                        for loc, tags in cityMap.tags.items()
                        if endTag in tags
                    ]
                # Else, return all the successors of current location and their corresponding distances according to the cityMap
                return [
                    (
                        neighbour,
                        State(location=neighbour, memory=None),
                        float(distance),
                    )
                    for neighbour, distance in cityMap.distances[state.location].items()
                ]
                # ### END CODE HERE ###

        # Call UCS.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`isEnd` always returns False), will exhaustively
        # compute costs to *all* other states.
        # ### START CODE HERE ###
        ucs = UniformCostSearch(verbose=0)
        #Seed search from the fake end node to propagate the cheapest cost. 
        ucs.solve(ReverseShortestPathProblem())
        # ### END CODE HERE ###

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `endTag`), we can retrieve `ucs.pastCosts`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!
        # ### START CODE HERE ###
        self.costs = ucs.pastCosts
        # ### END CODE HERE ###

    def evaluate(self, state: State) -> float:
        # ### START CODE HERE ###
        return self.costs.get(state.location, float("inf"))
        # ### END CODE HERE ###
