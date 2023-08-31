# State

## Intent

Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.

## Motivation

Consider a class TCPConnection that represents a network connection. A TCPConnection object can be in on of several different states: Established, Listening, Closed. When a TCPConnection object receives requests from other objects, it responds differently depending on its current state.

The State pattern describes how TCPConnection can exhibit different behavior in each state.

The TCPState class declares an interface common to all classes that represent different operational states.

The class TCPConnection maintains a state object that represents the current state of the TCP connection. The TCPConnection delegates all state-specific requests to this state object. TCPConnection uses its TCPState subclass instance to perform operations particular to the state of connection.

<img src="./img/state_motivation.svg" title="" alt="state_motivation!" data-align="center">

## Applicability

- An object's behavior depends on its state, and it must change its behavior at run-time depending on that state

- Often, several operations will contain this same conditional structure. The State patter puts each branch of conditional in a separate class

## Structure

<img src="./img/state_structure.svg" title="" alt="state_structure!" data-align="center">

## Collaborations

- Context delegates state-specific requests to the current ConcreteState object

- A context may pass itself as an argument to the State object handling the request. This lets the State object access the context if necessary

- Client can configure a context with State objects, Once a context is configured, its clients don't have to deal with the State objects directly

- Either Context or the ConcreteState subclasses can decide which state succeeds another and under what circumstances

## Consequences

1. *It localizes state-specific behavior and partitions behavior for different states*.
   
   - The State pattern puts all behavior associated with a particular state into one object.
   
   - This increases the numb of classes and is less compact than a single class. But such distribution is actually good if there are many states, which would otherwise necessitate large conditional statements (Like long procedures, large conditional statements are undesirable. They're monolithic and tend to make the code less explicit, which in turn makes them difficult to modify and extend)

2. *It makes state transitions explicit*
   
   - Introduce separate objects for different states makes the transitions more explicit

3. *State objects can be shared*
   
   - States are essentially flyweights with no intrinsic state, on;y behavior

## Implementation

1. *Who defines the state transitions?*

2. *A table-based alternative*
   
   - Advantage: You can change the transition criteria by modifying data instead of changing program code
   
   - Disadvantages:
     
     - Table look-up is often less efficient
     
     - Tabular format makes transition criteria harder to understand
     
     - Difficult to add actions to accompany the state transitions
   
   - The State pattern models state-specific behavior, whereas the table-driven approach focuses on defining state transitions

3. *Creating and destroying State objects*
   
   - Create State objects only when they are needed and destroy them thereafter (contexts change state infrequently)
   
   - Creating them ahead of time and never destroying them (changes occur rapidly)
