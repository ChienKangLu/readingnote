# Functions

## Small!

The first rule of functions is that they should be small.

## Blocks and Indenting

- Functions should not be large enough to hold nested structures.

- Indent level of a function should not be greater than one or two.

## Do One Thing

- ***FUNCTIONS SHOULD DO ONE THING. THEY SHOULD DO IT WELL. THEY SHOULD DO IT ONLY.***

- The reason we write functions is to decompose a larger concept (in other words, the name of function) into a set of setups at the next level of abstraction.

- Functions that do one thing cannot be reasonably divided into sections.

## One Level of Abstraction per Function

- In order to make sure our functions are doing "one thing", we need to make sure that the statements within our function are all at the same level of abstraction.

- Mixing levels of abstraction within a function is always confusing.

## Reading Code from Top to Bottom: The *Stepdown Rule*

- We want the code to read like a top-down narrative.

- Making the code read like a top-down set of *TO* paragraphs is an effective technique for keeping the abstraction level consistent.

## Switch Statements

```java
public Money calculatePay(Employee e) throws InvalidEmployeeType {
    switch(e.type) {
        case COMMISSIONED:
            return calculateCommissionedPay(e);
        case HOURLY:
            return calculateHourlyPay(e);
        case SALARIED:
            return calculateSalariedPay(e);
        default:
            throw new InvalidEmployeeType(e.type);
    }
}
```

Above snippet violate:

- Too large

- Do more than one thing

- Violate the Single Responsibility Principle (SRP): Because there is more than one reason for it to change

- Violate the Open Closed Principle (OCP): Because it must change whenever new types are added

- Unlimited number of other functions that will have the same structure

The solution to this problem is to bury `switch` statement in the basement of an ABSTRACT FACTORY, and never let anyone see it.

```java
public abstract class Employee {
    public abstract boolean isPayday();
    public abstract Money calculatePay();
    public abstract void deliverPay(Money pay);
}


public interface EmployeeFactory {
    public Employee makeEmployee(EmployeeRecord r) throws InvalidEmployeeType;
}

public class EmployeeFactoryImpl implements EmployeeFactory {
    public Employee makeEmployee(EmployeeRecord r) throws InvalidEmployeeType {
        switch(e.type) {
        case COMMISSIONED:
            return CommissionedEmployee(r);
        case HOURLY:
            return HourlyEmployee(r);
        case SALARIED:
            return SalariedEmployee(r);
        default:
            throw new InvalidEmployeeType(r.type);
       }
    }
}
```

## Use Descriptive Names

- It is hard to overestimate the value of good names.

- "*You know you are working on clean code when each routine turns out to be pretty much what you expected.*"

- The smaller and more focused a function is, the easier it is to choose a descriptive name.

- Don't be afraid to make a name long. A long name is better than a short enigmatic name.

- A long descriptive name is better than a long descriptive comment.

- Don't afraid to spend time choosing a name. Indeed, you should try several different names and read the code with each in place.

- Choosing descriptive names will clarify the design of the module in your mind and help you to improve it.

- Be consistent in your names. Use the same phrases, names, and verbs in the function names you choose for your modules.

## Function Arguments

- The ideal number of arguments for a function is zero (niladic). Next comes one (monadic), followed closely by two (dyadic). Three arguments (triadic) should be avoided where possible

## Common Monadic Forms

Common reasons to pass a single argument into a function:

- Asking a question about the argument
  `boolean fileExists("MyFile")`

- Operating on that argument, transforming it into something else and *returning it*.
  `InputStream fileOpen("MyFile")`

- *Event*. In this form there is an input argument but no output argument.
  `void passwordAttemptFailedNtimes(int attempts)`

- Using an output argument instead of a return value for a transformation is confusing.

## Flag Arguments

- Flag arguments are ugly. Passing a bolean into a function is truly terrible practice because it loudly proclaiming that this function does more than one thing.
