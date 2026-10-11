"""Gap de pre-mercado del ES como **dato declarado** del gate (regla 21, tarea #148).

La pregunta que responde este modulo: **que hace el gate cuando el modelo y el futuro se
contradicen?** La respuesta es la **regla 21**: la sesion se degrada a ``NOTHING`` —un bloqueo
justificado, nunca una direccion nueva—.

El dato lo **mide** `analysis/premarket_gap.py` (#141) y llega aqui **ya calculado**: este modulo
no descarga, no lee el disco y no mira el reloj. Solo lo tipa, lo valida y declara si va en contra
de la direccion del modelo.

Por que el dato vive en `decision/` y no se importa de `analysis/`
-----------------------------------------------------------------
`analysis/pipeline_report.py` importa `decision.gate`; si el gate importara
`analysis.premarket_gap`, el ciclo romperia la importacion. El dato se copia aqui con su
**vocabulario** (los estados de #141) y el llamante lo construye con
:meth:`PremarketDecision.from_payload`, que es el **unico** punto de conversion. Es el mismo patron
que ``OverlayDecision`` (#35).

El umbral
---------
``CONFLICT_THRESHOLD_PCT`` es una **decision del propietario** pre-registrada en `plan.md` §19.23
(2026-10-11): **0,25 %**. Esta por debajo de ``R`` (1 %, §19.12) y por encima del coste declarado
de ida y vuelta (~0,20 %, §3.3): un pre-mercado que ya mueve mas que el coste no es ruido. Con ese
umbral, la sesion del **2026-10-09** (`overnight_move_pct = +0,291 %` contra un modelo ``SHORT``)
**habria bloqueado**. Esto **no** es una afirmacion de *edge*: la ejecucion sigue siendo manual.

Los tres numeros no se suman
----------------------------
La regla 21 compara **un** numero, ``overnight_move_pct`` (el movimiento del **propio** futuro,
sin el *basis* estructural). ``gap_vs_index_pct`` y ``basis_pct`` se conservan porque son el
contrato de #141 y el informe los publica, pero **no** entran en la comparacion.

Este modulo es **puro**: biblioteca estandar, Pydantic v2 y el vocabulario de direcciones de
`backtest.engine`. Sin red, sin disco, sin reloj.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cfdtrader.backtest.engine import Direction

__all__ = [
    "CONFLICT_THRESHOLD_PCT",
    "PremarketDecision",
    "PremarketError",
    "PremarketState",
]

#: Umbral del propietario (§19.23): por debajo de ``R`` y por encima del coste declarado.
CONFLICT_THRESHOLD_PCT: Final[Decimal] = Decimal("0.25")


class PremarketState(StrEnum):
    """Los estados que publica #141. Se **consumen** tal cual: aqui no se inventa vocabulario."""

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"
    NOT_A_SESSION = "not_a_session"


class PremarketError(ValueError):
    """Un dato de pre-mercado inadmisible: un estado desconocido o un numero que no lo es."""


def _decimal_or_none(value: object, *, field: str) -> Decimal | None:
    """El numero **por su texto** (``Decimal(str(v))``): la unica conversion exacta de un float.

    ``None`` es una ausencia declarada y se conserva como ``None``; un valor no numerico o no
    finito es error tipado. Un ``bool`` no es un numero (``True`` no es ``1 %``).
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PremarketError(
            f"{field}: se espera un numero o `None`, no {type(value).__name__} ({value!r})"
        )
    number = Decimal(str(value))
    if not number.is_finite():
        raise PremarketError(f"{field}: se espera un numero finito, llego {value!r}")
    return number


class PremarketDecision(BaseModel):
    """El gap del pre-mercado **ya calculado**: el gate solo lo lee y lo compara.

    ``state`` y los tres numeros son los de #141. La regla 21 usa **solo**
    ``overnight_move_pct``; los otros dos viajan para que el informe publique el contrato
    completo (los tres numeros nunca se suman).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: date = Field(description="sesion (ET) a la que pertenece el dato medido")
    state: PremarketState = Field(description="`measured`, `unavailable` o `not_a_session`")
    overnight_move_pct: Decimal | None = Field(
        default=None,
        description=(
            "ES(pre-mercado)/ES(cierre t-1)-1 en %: el **unico** numero que compara la regla 21"
        ),
    )
    gap_vs_index_pct: Decimal | None = Field(
        default=None, description="ES(pre-mercado)/^GSPC(cierre t-1)-1 en %: incluye el *basis*"
    )
    basis_pct: Decimal | None = Field(
        default=None, description="ES(cierre t-1)/^GSPC(cierre t-1)-1 en %: el *basis* estructural"
    )
    reason: str = Field(description="motivo declarado: de la medida o de la ausencia")

    @model_validator(mode="after")
    def _check_medida_y_ausencia(self) -> PremarketDecision:
        """Un ``measured`` **tiene** su numero y una ausencia **no** inventa un ``0``."""
        move = self.overnight_move_pct
        if self.state is PremarketState.MEASURED:
            if move is None:
                raise ValueError(
                    "state = 'measured' sin `overnight_move_pct`: la medida esta incompleta"
                )
            return self
        if move is not None:
            raise ValueError(
                f"state = {self.state.value!r} con `overnight_move_pct` = {move}: una ausencia "
                "declarada no lleva numero, y menos un 0"
            )
        return self

    @property
    def measured(self) -> bool:
        """Si el dato esta **medido**: es la unica condicion que abre la regla 21."""
        return self.state is PremarketState.MEASURED

    def conflicts_with(self, direction: Direction) -> bool:
        """Si el pre-mercado va **en contra** de esa direccion, con magnitud suficiente.

        El umbral es ``CONFLICT_THRESHOLD_PCT`` y el numero comparado es **solo**
        ``overnight_move_pct``. Sin direccion (``NOTHING``) no hay contradiccion que medir, y sin
        medida (``None``) no hay nada que comparar: los dos casos devuelven ``False``.
        """
        move = self.overnight_move_pct
        if move is None or direction is Direction.NOTHING:
            return False
        if direction is Direction.LONG:
            return move <= -CONFLICT_THRESHOLD_PCT
        return move >= CONFLICT_THRESHOLD_PCT

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PremarketDecision:
        """Construye el dato declarado desde el payload de #141 (**unico** punto de conversion).

        Los tres numeros del payload llegan como numero decimal en coma flotante y se convierten
        **por su texto**: la aritmetica del gate no vuelve a ver un ``float``.
        """
        state = payload.get("state")
        if not isinstance(state, str):
            raise PremarketError(f"el payload de #141 no declara `state`: {state!r}")
        try:
            declared = PremarketState(state)
        except ValueError as error:
            raise PremarketError(f"estado de pre-mercado desconocido: {state!r}") from error
        session = payload.get("session")
        if not isinstance(session, str):
            raise PremarketError(f"el payload de #141 no declara `session`: {session!r}")
        try:
            day = date.fromisoformat(session)
        except ValueError as error:
            raise PremarketError(f"`session` no es una fecha ISO-8601: {session!r}") from error
        reason = payload.get("reason")
        if not isinstance(reason, str):
            raise PremarketError(f"el payload de #141 no declara `reason`: {reason!r}")
        try:
            return cls(
                session=day,
                state=declared,
                overnight_move_pct=_decimal_or_none(
                    payload.get("overnight_move_pct"), field="overnight_move_pct"
                ),
                gap_vs_index_pct=_decimal_or_none(
                    payload.get("gap_vs_index_pct"), field="gap_vs_index_pct"
                ),
                basis_pct=_decimal_or_none(payload.get("basis_pct"), field="basis_pct"),
                reason=reason,
            )
        except ValidationError as error:
            raise PremarketError(f"dato de pre-mercado inadmisible: {error}") from error
