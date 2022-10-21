# Composite

## Intent

- Compose objects into tree structure.

- Treat individual objects and compositions of objects uniformly.

## Motivation

- Composite patter describes how to use recursive composition so that clients don't have to make the distinction.

- The key is an abstract class that represents both primitives and their containers.

## Applicability

- Represent part-while hierarchies of objects.

- Treat all objects in the composite structure uniformly.

## Structure

![composite!](./img/composite.svg)

## Implementation

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
