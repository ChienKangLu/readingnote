# Bridge

## Intent

Decouple an abstraction from its implementation so that the two can vary independently.

## Motivation

- Put abstraction and implementation in **separate class hierarchies**.

- All operations on abstractions are implemented in terms of abstract operations from implementor.

- We refer to the relationship between `Abstraction` and `Implementor` as **bridge**, because it bridges the abstraction and its implementation, letting them vary independently.

## Applicability

- Nested generalizations.

- The implementation must be selected or switched at run-time.

- The abstractions and implementations should be extensible by subclassing.

- Hide the implementation.

## Structure

![bridge!](./img/bridge.svg)

## Participants

Typically the `Implementor` interface provides only primitive operations, and `Abstraction` defines higher-level operations based on these primitives.

## Consequences

- The implementation of an abstraction can be configured at run-time.

- This decoupling encourages layering that can lead to a better-structured system. The high-level part of system only has to know `Abstraction` and `Implementor`.

- Hiding implementation detail from client.

## Implementation

Only one `Implementor`

- A degenerate case of Bridge, one-to-one relationship between `Abstraction` and `Implementor`.

Creating the right `Implementor`

- Instantiate in `Abstraction`'s constructor.

- Decide by parameter passed to its constructor.

- Default implementation initially.

- Delegate the decision to another objects, factory object.
