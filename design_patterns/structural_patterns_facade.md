# Facade

## Intent

- Provide a unified interface to a set of interface in subsystem.

- Facade defines a higher-level interface that makes subsystem easier to use.

## Motivation

- Minimize the communication and dependencies between subsystems.

- A facade object provides a single, simplified interface to the more general facilities of subsystem.

- The powerful but low-level interfaces in subsystem only complicate their task.

## Applicability

- Provide a simple interface to a complex subsystem.

- Introduce a facade to decouple the subsystem from clients and other subsystems, thereby promoting subsystem independence and portability.

- Layer subsystems.

- Use a facade to define an entry point to each subsystem level. If subsystems are dependent, simplify the dependencies between them by making them communicate with each other solely through their facade.

## Structure

![facade!](./img/facade.svg)

## Collaborations

- Clients communicate with subsystem  by sending requests to Facade.

- Clients don't have to access its subsystem objects directly.

## Consequences

- It shields clients from subsystem components.

- Makes the subsystem easier to use.

- It promotes weak coupling between the subsystem and its clients.

- It eliminates complex or circular dependencies.

- Reducing compilation dependencies.

- It doesn't prevent applications from using subsystem classes if they need to.

## Implementation

- Reducing client-subsystem coupling
  
  - Subclassing for different implementation of a subsystem.
  
  - Configure with different subsystem objects.

- Facade class is part of the public interface.

- Making subsystem classes private would be useful.
