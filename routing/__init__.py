from abc import ABC, abstractmethod

from optmod.routing.context import RoutingContext
from optmod.schemas import RoutingDecision


class BaseRouter(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Must never raise. Catch all exceptions internally."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _passthrough(self, ctx: RoutingContext, reason: str) -> RoutingDecision:
        from optmod.schemas import RoutingDecision
        return RoutingDecision(
            model=ctx.registry.primary,
            mutator="noop",
            reason=reason,
            confidence=1.0,
            router_name=self.name,
        )


def build_router(name: str, config: dict) -> BaseRouter:
    from .passthrough        import PassthroughRouter
    from .rule_based         import RuleBasedRouter
    from .decision_tree      import DecisionTreeRouter
    from .trouter               import TRouterRouter
    from .perfrouter          import PerfRouterRouter

    mapping = {
        "passthrough":   PassthroughRouter,
        "rule_based":    RuleBasedRouter,
        "decision_tree": DecisionTreeRouter,
        "trouter":       TRouterRouter,
        "perf_router":   PerfRouterRouter,
    }
    cls = mapping.get(name, PassthroughRouter)
    return cls(config)
