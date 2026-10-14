# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""YAML-targeted optimizer builders: AdamW/Muon selection.

Parameter-name and parameter-group logic lives in
:mod:`hyper_parallel.components.optim.parameter_groups`; the optimizer
algorithm implementations stay in ``hyper_parallel.core.optimizer``.
"""

__all__ = ["AdamW", "Muon"]

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.components.optim.parameter_groups import (
    _DEFAULT_ADAMW_NAME_KEYWORDS,
    get_adamw_param_groups,
    split_muon_adamw_params,
)

logger = logging.getLogger(__name__)

# Opt-in switch that reproduces MindFormers' AdamW grouping: every AdamW parameter decays,
# including the 1-D norm and bias parameters.  Off by default, HP's historical behaviour.
WD_ALL_PARAMS_KEY = "adamw_wd_all_params"


def _adamw_optimizer_kwargs(adamw_config: dict) -> Dict[str, Any]:
    """Return the AdamW config without the grouping-only switch."""
    return {name: value for name, value in adamw_config.items() if name != WD_ALL_PARAMS_KEY}


def _keep_declared_weight_decay(optimizer: Any) -> None:
    """Keep the configured AdamW weight decay across a checkpoint restore.

    A checkpoint stores its parameter groups verbatim and ``load_state_dict`` installs them, so a
    restored optimizer would silently revert the grouping the configuration asked for (see
    :func:`hyper_parallel.components.optim.parameter_groups.get_adamw_param_groups`).  The declared
    values are captured here and re-applied by a load-state post-hook.

    Args:
        optimizer: Optimizer or chained optimizer returned by the hyper optimizer factory.
    """
    leaf = getattr(optimizer, 'optimizers_dict', {}).get('adamw')
    if leaf is None:
        return
    declared = [float(param_group['weight_decay']) for param_group in leaf.param_groups]

    def reapply_weight_decay(module, *args, **kwargs):  # pylint: disable=unused-argument
        """Put the declared weight decay back after the checkpoint's groups were installed."""
        for param_group, weight_decay in zip(module.param_groups, declared):
            param_group['weight_decay'] = weight_decay

    leaf.register_load_state_dict_post_hook(reapply_weight_decay)


class AdamW:
    """Build a core AdamW optimizer from YAML configuration."""

    def __init__(
            self,
            adamw_config: dict,
            model: nn.Module,
            no_decay_params: Optional[List[str]] = None,
    ) -> None:
        """Initialize AdamW optimizer configuration.

        Args:
            adamw_config: AdamW hyperparameters resolved from YAML.
            model: Module whose trainable parameters are optimized.
            no_decay_params: Optional names excluded from weight decay.
        """
        self.config = adamw_config
        self.model = model

        adamw_groups, _ = self.get_adamw_param_groups(
            self.model,
            weight_decay=adamw_config.get("adamw_weight_decay", 1e-2),
            no_decay_params=no_decay_params,
            decay_all_params=bool(adamw_config.get(WD_ALL_PARAMS_KEY, False)),
        )
        if not adamw_groups:
            raise ValueError("AdamW requires at least one trainable parameter")

        self.optimizer = get_hyper_optimizer(
            model=self.model,
            muon_params=[],
            adamw_params=adamw_groups,
            adamw_kwargs=_adamw_optimizer_kwargs(adamw_config),
        )
        if adamw_config.get(WD_ALL_PARAMS_KEY, False):
            _keep_declared_weight_decay(self.optimizer)

    @staticmethod
    def get_adamw_param_groups(
            model: "nn.Module",
            weight_decay: float = 1e-2,
            no_decay_params: Optional[Sequence[str]] = None,
            param_groups: Optional[Sequence[Dict[str, Any]]] = None,
            allowed_param_ids: Optional[Sequence[int]] = None,
            decay_all_params: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Split model parameters into decaying and non-decaying groups.

        Delegates to
        :func:`hyper_parallel.components.optim.parameter_groups.get_adamw_param_groups`;
        retained on the class so existing call sites keep working.

        Args:
            model: Module whose trainable parameters are routed.
            weight_decay: Default decay coefficient for trainable parameters.
            no_decay_params: Name keywords excluded from weight decay.
            param_groups: Optional prebuilt groups to return unchanged.
            allowed_param_ids: Optional parameter identities eligible for routing.
            decay_all_params: Whether every routed parameter decays, including the 1-D norm
                and bias parameters MindFormers also decays.

        Returns:
            Parameter groups and names routed to AdamW.
        """
        return get_adamw_param_groups(
            model,
            weight_decay=weight_decay,
            no_decay_params=no_decay_params,
            param_groups=param_groups,
            allowed_param_ids=allowed_param_ids,
            decay_all_params=decay_all_params,
        )

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer


class Muon:
    """Build a mixed Muon and AdamW optimizer from YAML configuration."""

    _DEFAULT_ADAMW_NAME_KEYWORDS = _DEFAULT_ADAMW_NAME_KEYWORDS

    def __init__(
            self,
            muon_config: dict,
            adamw_config: dict,
            model: nn.Module,
            extra_adamw_name_keywords: Optional[List[str]] = None,
            no_decay_params: Optional[List[str]] = None,
    ) -> None:
        """Build a mixed Muon and fallback AdamW runtime for ``model``.

        Args:
            muon_config: Muon hyperparameters resolved from YAML.
            adamw_config: Fallback AdamW hyperparameters resolved from YAML.
            model: Module whose trainable parameters are optimized.
            extra_adamw_name_keywords: Additional names routed to AdamW.
            no_decay_params: Optional names excluded from weight decay.
        """
        self.muon_config = muon_config
        self.adamw_config = adamw_config
        self.model = model

        muon_params, adamw_params, muon_names, adamw_names = self.split_muon_adamw_params(
            model,
            extra_adamw_name_keywords=extra_adamw_name_keywords or (),
        )
        if not muon_params:
            raise ValueError("Muon requires at least one eligible matrix parameter")

        adamw_groups, _ = AdamW.get_adamw_param_groups(
            model,
            weight_decay=adamw_config.get("adamw_weight_decay", 1e-2),
            no_decay_params=no_decay_params,
            allowed_param_ids=[id(parameter) for parameter in adamw_params],
            decay_all_params=bool(adamw_config.get(WD_ALL_PARAMS_KEY, False)),
        )

        logger.info_rank0(
            "Muon optimizer split: %s Muon parameters, %s AdamW parameters",
            len(muon_names),
            len(adamw_names),
        )
        logger.info_rank0("Muon parameters (first 5): %s", muon_names[:5])
        logger.info_rank0("AdamW parameters (first 5): %s", adamw_names[:5])

        self.optimizer = get_hyper_optimizer(
            model=self.model,
            muon_params=muon_params,
            adamw_params=adamw_groups,
            muon_kwargs=muon_config,
            adamw_kwargs=_adamw_optimizer_kwargs(adamw_config),
        )
        if adamw_config.get(WD_ALL_PARAMS_KEY, False):
            _keep_declared_weight_decay(self.optimizer)

    @staticmethod
    def split_muon_adamw_params(
            model: nn.Module,
            extra_adamw_name_keywords: Sequence[str] = (),
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[str], List[str]]:
        """Route matrix parameters to Muon and remaining parameters to AdamW.

        Delegates to
        :func:`hyper_parallel.components.optim.parameter_groups.split_muon_adamw_params`;
        retained on the class so existing call sites keep working.

        Args:
            model: Module whose trainable parameters are routed.
            extra_adamw_name_keywords: Additional names reserved for AdamW.

        Returns:
            Muon parameters, AdamW parameters, and their respective names.
        """
        return split_muon_adamw_params(
            model,
            extra_adamw_name_keywords=extra_adamw_name_keywords,
        )

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer
