# Abstract Factory

## Intent

- Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
- Factory of factory

## Motivation

Consider a user interface that supports multiple look-and feel. Different look-and-feel define different appearances. Instantiating loo-and-feel-specific classes of widgets throughout the application makes it hard to change the look and feel later.

## Applicability

- a system should be independent of how its products are created, composed and represented

- a system should be configured with one of multiple families of products

- a family of related product objects its designed to used together, and you need to enforce this constraint

- you want to provide a class library of products, and you want to reveal just their interfaces, not their implementations.

## Structure

![abstract_factory_structure!](./img/abstract_factory_structure.svg)

## Consequences

- It isolates clients from concrete classes

- It makes exchanging product families easy

- It promotes consistency among products

- Supporting new kinds of products is difficult

## Implementation

- Factories as singletons: An application typically needs only one instance of a ConcreteFactory per product family.
