"""Reserva del *holdout* final intocable (`plan.md` §11.4 y §21 pregunta 10) — tarea #68.

La pregunta que responde este modulo: **que sesiones no se pueden tocar hasta la decision
final de ir a produccion, y quien lo impide?**

La decision, declarada
----------------------

`plan.md` §11.4 pide reservar un periodo y **no mirarlo** hasta el unico vistazo final;
§21 pregunta 10 pide el periodo exacto y `tech_stack.md` §11 bis (decision 7) exige fijarlo
**antes** de ver resultados. Es una decision del **propietario**, y aqui vive como
:data:`HOLDOUT_DECISION`: un literal con el tramo, su motivo y su fecha, no una medicion.
Una decision del propietario se anota en el documento que la declaraba abierta
(`plan.md` §11.4 y §21 pregunta 10), y este modulo la aplica.

El mecanismo, y por que falla en voz alta
-----------------------------------------

- :func:`reserve` parte una secuencia de sesiones en **(usable, reservado)**. Comprueba que
  el tramo reservado sea **el final** de la muestra y que tenga **exactamente** las sesiones
  que la decision declara: si el universo crece o encoge, la reserva **no** se recoloca sola
  —un *holdout* que se mueve con la muestra no reserva nada— y el desajuste es un error tipado.
- :func:`require_usable` es la puerta: recibe las sesiones que un consumidor va a usar y
  **falla** si alguna cae en el tramo. Documentarlo no basta (A3): el codigo de evaluacion
  tiene que **chocar** contra el.
- :func:`holdout_report` publica el recuento, el rango y que queda fuera de los folds (A4).

Lo que este modulo **no** hace
-----------------------------

- **No** genera particiones: eso es #12 (`backtest.splits`), que recibe este tramo ya
  apartado y nunca ve las sesiones reservadas.
- **No** entrena, calibra ni evalua (**#13**, **#24**, **#25**).
- **No** lee el `Store`, `config/` ni `data/`: es puro respecto al disco, **no** consulta el
  reloj (el "hoy" es un campo declarado de la decision) y no usa azar.
- **No** decide el periodo: lo aplica. Cambiarlo es otra decision del propietario, y
  cambiarlo **despues** de haber mirado el resultado seria mover la porteria (`plan.md` §11.6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Final

__all__ = [
    "HOLDOUT_DECISION",
    "HOLDOUT_DOES_NOT_DO",
    "HoldoutAccessError",
    "HoldoutDecision",
    "HoldoutError",
    "HoldoutMismatchError",
    "ReservedSample",
    "holdout_report",
    "require_usable",
    "reserve",
]


@dataclass(frozen=True, slots=True)
class HoldoutDecision:
    """La decision del propietario: que tramo se reserva, por que y desde cuando.

    Es un **literal declarado**, no una medicion: por eso ``n_sessions`` viaja con el rango
    —para poder comprobar que la reserva cuadra con el universo real— y ``decided_on`` fecha
    la decision, que es lo que permite demostrar que se tomo antes de ver el resultado.
    """

    first_session: date
    last_session: date
    n_sessions: int
    reason: str
    decided_on: date
    decided_by: str

    def contains(self, session: date) -> bool:
        """``True`` si esa sesion cae dentro del tramo reservado."""
        return self.first_session <= session <= self.last_session


#: **La decision del propietario (2026-10-06, issue #68).** Los ultimos **12 meses** del
#: universo etiquetado, que es el ejemplo de `plan.md` §11.4, medido sobre `derived.labels`
#: (2.688 sesiones de 2016-01-07 a 2026-09-17).
#:
#: Se fija **antes** de mirar el resultado, como exige `tech_stack.md` §11 bis: cambiar el
#: tramo despues seria mover la porteria (`plan.md` §11.6).
HOLDOUT_DECISION: Final[HoldoutDecision] = HoldoutDecision(
    first_session=date(2025, 9, 18),
    last_session=date(2026, 9, 17),
    n_sessions=251,
    reason=(
        "los **ultimos 12 meses** del universo etiquetado (el ejemplo de `plan.md` §11.4): "
        "un ano de sesiones reales basta para el unico vistazo final y deja el walk-forward "
        "con toda su cola anterior"
    ),
    decided_on=date(2026, 10, 6),
    decided_by="propietario (issue #68)",
)

#: Que **no** hace este modulo, legible por maquina. Cada frontera con su issue.
HOLDOUT_DOES_NOT_DO: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "no_genera_particiones",
        "issue": "#12",
        "statement": (
            "no genera particiones walk-forward: recibe la secuencia y entrega el tramo "
            "reservado ya apartado, para que #12 no lo vea nunca"
        ),
    },
    {
        "id": "no_entrena_ni_evalua",
        "issue": "#24",
        "statement": "no entrena, calibra ni evalua: no toca ningun modelo ni metrica",
    },
    {
        "id": "no_lee_disco_ni_reloj",
        "issue": "#68",
        "statement": (
            "no lee el Store, config/ ni data/, y no consulta el reloj: el 'hoy' de la "
            "decision es un campo declarado"
        ),
    },
)


class HoldoutError(Exception):
    """Raiz de los errores de la reserva del *holdout*."""


class HoldoutMismatchError(HoldoutError):
    """La reserva no cuadra con el universo: el tramo no es el final, o no tiene sus sesiones."""


class HoldoutAccessError(HoldoutError):
    """Alguien intento usar sesiones del tramo reservado (A3): se para, no se avisa."""


@dataclass(frozen=True, slots=True)
class ReservedSample:
    """La muestra partida: lo que se puede usar y lo que **no** se toca hasta el final."""

    usable: tuple[date, ...]
    reserved: tuple[date, ...]


def reserve(
    sessions: Sequence[date], *, decision: HoldoutDecision = HOLDOUT_DECISION
) -> ReservedSample:
    """Parte ``sessions`` en ``(usable, reservado)`` segun la decision declarada.

    ``sessions`` tiene que venir ordenada de forma **estricta** (sin repetidos), que es la
    unica forma de que cuenta y rango digan lo mismo. El tramo reservado tiene que ser el
    **final** de la muestra y tener **exactamente** ``decision.n_sessions`` sesiones: si no,
    es :class:`HoldoutMismatchError`. La reserva no se recoloca sola.

    Returns
    -------
    ReservedSample
        ``usable`` (todo lo anterior al tramo) y ``reserved`` (el tramo), sin solapes.
    """
    if not sessions:
        raise HoldoutMismatchError("la secuencia de sesiones esta vacia: no hay nada que reservar")
    for previous, current in pairwise(sessions):
        if current <= previous:
            raise HoldoutMismatchError(
                "las sesiones tienen que ser estrictamente crecientes y sin repetidos: "
                f"{previous.isoformat()} -> {current.isoformat()}"
            )

    reserved = tuple(session for session in sessions if decision.contains(session))
    usable = tuple(session for session in sessions if not decision.contains(session))

    if not usable or not reserved:
        raise HoldoutMismatchError(
            "la particion se comeria la muestra entera: no queda muestra usable o no queda "
            "nada que reservar"
        )
    if len(reserved) != decision.n_sessions:
        raise HoldoutMismatchError(
            f"la decision reserva {decision.n_sessions} sesiones y el universo tiene "
            f"{len(reserved)} entre {decision.first_session.isoformat()} y "
            f"{decision.last_session.isoformat()}: la reserva no se recoloca sola, se decide"
        )
    if reserved[0] != decision.first_session or reserved[-1] != decision.last_session:
        raise HoldoutMismatchError(
            "el tramo reservado no coincide con el rango declarado "
            f"({decision.first_session.isoformat()} -> {decision.last_session.isoformat()})"
        )
    if usable[-1] > decision.last_session:
        raise HoldoutMismatchError(
            "el holdout tiene que ser el **final** de la muestra: hay sesiones usables "
            f"posteriores a {decision.last_session.isoformat()}"
        )
    return ReservedSample(usable=usable, reserved=reserved)


def require_usable(
    sessions: Sequence[date], *, decision: HoldoutDecision = HOLDOUT_DECISION
) -> tuple[date, ...]:
    """Devuelve ``sessions`` si **ninguna** cae en el tramo reservado; si cae, error tipado.

    Es la puerta del A3: no basta con documentar el *holdout*, el codigo de evaluacion tiene
    que chocar contra el. El mensaje nombra las sesiones ofensoras para que se vea que la
    comprobacion es real.
    """
    offenders = tuple(session for session in sessions if decision.contains(session))
    if offenders:
        preview = ", ".join(session.isoformat() for session in offenders[:3])
        raise HoldoutAccessError(
            f"{len(offenders)} sesion(es) caen en el holdout final intocable "
            f"({decision.first_session.isoformat()} -> {decision.last_session.isoformat()}): "
            f"{preview}. El tramo solo se mira una vez, en la decision de ir a produccion "
            "(`plan.md` §11.4)"
        )
    return tuple(sessions)


def holdout_report(
    sessions: Sequence[date], *, decision: HoldoutDecision = HOLDOUT_DECISION
) -> dict[str, object]:
    """Payload que declara la reserva: decision, recuento, rango y que queda fuera (A4).

    ``sessions`` es el universo etiquetado (el mismo que recibe #12). El informe **declara**
    el tramo y el hueco que deja; no lo recalcula ni lo mueve.
    """
    sample = reserve(sessions, decision=decision)
    return {
        "decision": {
            "first_session": decision.first_session.isoformat(),
            "last_session": decision.last_session.isoformat(),
            "n_sessions": decision.n_sessions,
            "reason": decision.reason,
            "decided_on": decision.decided_on.isoformat(),
            "decided_by": decision.decided_by,
        },
        "n_sessions_total": len(sessions),
        "n_reserved": len(sample.reserved),
        "reserved_range": [sample.reserved[0].isoformat(), sample.reserved[-1].isoformat()],
        "n_usable": len(sample.usable),
        "usable_range": [sample.usable[0].isoformat(), sample.usable[-1].isoformat()],
        "excluded_from_folds": (
            "el tramo reservado queda fuera de **todos** los folds de walk-forward: #12 "
            "recibe solo `usable`, y `require_usable` impide que se cuele"
        ),
        "note": (
            "la reserva es una decision del propietario, no una medicion: se fija antes de "
            "mirar el resultado y no se recalcula con la muestra"
        ),
    }
