# Adapter

## Intent

- Convert the interface of a class into another interface clients expect.

- What if you want to use a class that belongs to a third party app, but it doesn't fit with your app because of incompatible interfaces. The adapter pattern provides a way for classes to work together that normally wouldn't be able to.

## Motivation

We could define `TextShape` (adapter) so that it *adapts* the `TextView` interface to to `Shape`'s.

## Collaborations

Clients call operations on an Adapter instance. In turn, the adapter calls Adaptee operations that carry out the request.

`TextShape` (adapter) adapts `TextView` (adaptee or adapted class) to the Shape (target) interface.

`TextShape` adds the functionality that `TextView` lacks but Shape requires.

## Consequences

Class version: Inheriting `Shape`'s interface and `TextView`'s implementation.

![adapter_class_adapter!](./img/adapter_class_adapter.svg)

Object version: Composing a `TextView` instance within a `TextShape` and implementing `TextShape` in terms of `TextView`'s interface.

![adapter_object_adapter!](./img/adapter_object_adapter.svg)

*Pluggable adapter*:

- Build interface adaptation into classes.

- Describe classes with built-in interface adaptation.

- *Narrow* interface of Adaptee:
  
  - Using abstract operations
  
  - Using delegate objects: client can use different adaptation strategy by substituting a different delegate
  
  - Parameterized adapters
