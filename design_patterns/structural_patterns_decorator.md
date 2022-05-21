# Decorator

## Intent

Attach additional responsibilities to an object dynamically.

## Motivation

- Inheritance is a way to add responsibilities but client cannot control when and how to decorate component.

- A more flexible approach is to enclose the component in another object. The enclosing object is called decorator.

- The decorator forwards request to the component and may perform additional actions.

- Transparency lets you nest decorators recursively.

- The important aspect of this pattern is that it lets decorators appear anywhere a component can.

## Applicability

- To add responsibilities to individual objects dynamically and transparently.

- For responsibilities that can be withdrawn.

- When extension by subclassing is impractical.

## Structure

![decorator!](./img/decorator.svg)

## Consequences

- More flexibility than static inheritance
  
  - Responsibilities can be added and removed at run-time.
  
  - Providing different Decorator for specific Component lets you mix and match responsibilities.

- Avoid feature-laden classes high up in the hierarchy
  
  - Defines a simple class and add functionality incrementally.
  
  - Functionality can be composed from simple pieces.

- A decorator and its component aren't identical

- Lost of little objects

## Implementation

- Omitting the abstract Decorator class.

- Keeping Component classes lightweight.

- Changing the skin of an object versus changing its guts
  
  - We can think of a decorator as a skin over an object that changes its behavior.
  
  - The Strategy pattern is a good example of a pattern from changing the guts.
    
    - In situations where the Component class is intrinsically heavyweight.
    
    - Component forwards some of its behavior to a separate strategy object which can be altered.
    
    - Example: views maintain a list of "adorn" objects that can attach additional adornments like borders to a view component.
    
    - The component itself knows possible extensions for example, keyboard-handing support.