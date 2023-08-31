# Strategy

## Intent

Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

## Motivations

Defining classes that encapsulate different line-breaking algorithms. An Algorithm that's encapsulated in this way is called a **strategy**.

Suppose a **Composition** class is responsible for maintaining and updating the linebreaks of text displayed in a text viewer. Linebreaking strategies aren't implemented by the class **Composition**. Instead, they are implemented separately by subclasses of the abstract **Compositor** class. **Compositor** subclasses implement different strategies.

A **Composition** maintains a reference to a **Compositor** object. Whenever a **Composition** reformats its text, it forwards this responsibility to its **Compositor** object. The client of **Composition** specifies which **Compositor** should be used by installing the **Compositor** it desires into the **Composition**.

<img src="./img/strategy_motivation.svg" title="" alt="strategy_motivation!" data-align="center">

## Applicability

- many related classes differ only in their behavior. Strategies provide a way to configure a class with one of many **behaviors**

- you need different variants of an algorithm

- an algorithm uses data that clients shouldn't know about. Use the Strategy pattern to avoid exposing complex, algorithm-specific data structures

- a class defines many behaviors, and theses appear as multiple conditional statements in its operations

## Structure

<img src="./img/strategy_structure.svg" title="" alt="strategy_structure!" data-align="center">

## Collaborations

- Strategy and Context interact to implement the chosen algorithm. A context may pass all data required by the algorithm to strategy when the algorithm is called. Alternatively, the context can pass itself as an argument to Strategy operations. That lets the strategy call back on the context as required.

- A context forwards requests from its clients to its strategy. Clients usually create and pass a ConcreteStrategy object to the context; thereafter, clients interact with the context exclusively. There is often a family of ConcreteStrategy classes for a client to choose from.

## Consequences

1. *Families of related algorithms*

2. *An alternative to subclassing*
   
   - Inheritance offers another way to support a variety of algorithms or behaviors. Eventually, You wind up with many related classes whose only difference is the algorithm or behavior they employ.
   
   - Encapsulating the algorithm in separate Strategy classes lets you vary the algorithm independently of its context, making it easier to switch, understand, and extend

3. *Strategies eliminate conditional statements*. The Strategy pattern offers an alternative to conditional statements for selecting desired behavior
   
   - **Code containing many conditional statements often indicates the need to apply Strategy patter**

4. *A choice of implementations*. Strategy can provide different implementations of the same *behavior*

5. *Client must be aware of different Strategies*

6. *Communication overhead between Strategy and Context*. The Strategy interface is shared by all ConcreteStrategy classes whether the algorithm they implement are trivial or complex

7. *Increased number of objects*. Strategies increase the number of objects in an application. Sometimes you can reduce this overhead by implementing strategies as stateless objects that contexts can share.

## Implementation

1. *Defining the Strategy and Context interfaces*. The Strategy and Context interfaces must give a Concrete Strategy efficient access to any data it needs from a context, and vice versa
   
   1. Context pass data in parameters to Strategy operations.
   
   2. Context pass itself as an argument, and the strategy requests data from the context explicitly

2. *Strategies as template parameters*. This technique is only applicable if
   
   1. the Strategy can be selected at compile-time
   
   2. it does not have to be changes at run-time

3. *Making Strategy objects optional*. Context carries out default behavior
