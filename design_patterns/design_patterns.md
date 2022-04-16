# Structure Patterns

## Adapter

### Intent

Convert the interface of a class into another interface clients expect.

### Motivation

We could define `TextShape` (adapter) so that it *adapts* the `TextView` interface to to `Shape`'s.

### Collaborations

Clients call operations on an Adapter instance. In turn, the adapter calls Adaptee operations that carry out the request.

`TextShape` (adapter) adapts `TextView` (adaptee or adapted class) to the Shape (target) interface.

`TextShape` adds the functionality that `TextView` lacks but Shape requires.

### Consequences

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

## Bridge

### Intent

Decouple an abstraction from its implementation so that the two can vary independently.

### Motivation

- Put abstraction and implementation in **separate class hierarchies**.

- All operations on abstractions are implemented in terms of abstract operations from implementor.

- We refer to the relationship between `Abstraction` and `Implementor` as **bridge**, because it bridges the abstraction and its implementation, letting them vary independently.

### Applicability

- Nested generalizations.

- The implementation must be selected or switched at run-time.

- The abstractions and implementations should be extensible by subclassing.

- Hide the implementation.

### Structure

![bridge!](./img/bridge.svg)

### Participants

Typically the `Implementor` interface provides only primitive operations, and `Abstraction` defines higher-level operations based on these primitives.

### Consequences

- The implementation of an abstraction can be configured at run-time.

- This decoupling encourages layering that can lead to a better-structured system. The high-level part of system only has to know `Abstraction` and `Implementor`.

- Hiding implementation detail from client.

### Implementation

Only one `Implementor`

- A degenerate case of Bridge, one-to-one relationship between `Abstraction` and `Implementor`.

Creating the right `Implementor`

- Instantiate in `Abstraction`'s constructor.

- Decide by parameter passed to its constructor.

- Default implementation initially.

- Delegate the decision to another objects, factory object.

## Composite

#### Intent

- Compose objects into tree structure.

- Treat individual objects and compositions of objects uniformly.

#### Motivation

- Composite patter describes how to use recursive composition so that clients don't have to make the distinction.

- The key is an abstract class that represents both primitives and their containers.

#### Applicability

- Represent part-while hierarchies of objects.

- Treat all objects in the composite structure uniformly.

#### Structure

![composite!](./img/composite.svg)

#### Implementation

*Explicit parent reference*

- Simplify the traversal.

- Simplify moving up the structure and deleting a component .

*Maximizing the Component interface*

- it sometimes conflicts with the principle of class hierarchy design that says a class should only define operations that are meaningful to its subclass.

*Trade-off between safety and transparency*

- Transparency: define the child management interface at the root of the class hierarchy.

- Safety: define child management in `Composite` class.

*Putting the child pointer in the base class incurs a space penalty for every leaf*

*Child ordering*

*Caching to improve performance*

- Composite class can cache traversal or search information about its children.
  
  - Changes to a component will require invalidating the caches of its parents. You need to define an interface for telling composites that their caches are invalid.

*Who should delete component?*

- It's usually best to make a Composite responsible for deleting its children when it;s destroyed.

*The best data structure?*

- Composites may use a variety of data structures to store their children, including linked list, trees, arrays and has tables. The choice of data structure depends on efficiency.
