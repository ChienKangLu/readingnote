# Template Method

## Intent

Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure

## Motivation

A template method defines an algorithm in terms of abstract operations that subclasses override to provide concrete behavior.

By defining some of the steps of an algorithm using abstract operations, the template method fixes their ordering, but it lets subclasses vary those steps to suit their needs.

## Applicability

- to implement the invariant parts of an algorithm once and leave it up to subclasses to implement the behavior that can vary

- when common behavior among subclasses should be factored and localized in a common class to avoid code duplication. You first identify the differences in the existing code and then separate the differences into new operations. Finally you replace the differing code with a template method that calls one of these new operations

- to control subclasses extensions. You can define a template method that calls "hook" operations at specific points, thereby permitting extensions only at those points.

## Structure

<img src="./img/template_method_structure.svg" title="" alt="template_method_structure!" data-align="center">

## Collaborations

ConcreteClass relies on AbstractClass to implement the invariant steps of the algorithm.

## Consequences

Template methods are a fundamental technique for code reuse. They are particularly important in class libraries, because they are the means for factoring out common behavior in library class.

Template methods lead to an inverted control structure.

It's important for template methods to specify which operations are **hooks** (*may* be overridden) and which are **operations** (*must* be overridden).

## Implementation

1. *Using C++ access control*. In C++, the primitive operations that a template method calls can be declared *protected* members.
   
   - Primitive operations that *must* be overridden are declared pure virtual.
   
   - The template method itself should not be overridden

2. *Minimizing primitive operations*. Minimize the number of primitive operations that a subclass must override to flesh out the algorithm. The more operations that need overriding, the more tedious things get for clients

3. *Naming conventions*. Prefix template method names with "Do-".


