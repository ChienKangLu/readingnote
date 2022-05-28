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

## Common Monadic Forms (One argument)

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

## Dynamic Functions (Two arguments)

- A function with two arguments is harder to understand than a monadic function. `writeField(name)` us easier to understand than `writeField(output-Stream, name)`.

- Even obvious dyadic functions like `assertEquals(expected, actaul)` are problematic.

- There are times, of course, where two arguments are appropriate. For example, `Point p = new Point(0, 0)`

- You should be aware that they come at a cost and should take advantage of what mechanisms may be available to you to convert them into monads.

## Triads (Third arguments)

- Functions that take three arguments are significantly harder to understand than dyads. For example, `assertEquals(message, expected, actual)`

## Argument Objects

- When a function seems to need more than two or three arguments, it is likely that some of those arguments ought to be wrapped into a class of their own.

## Argument Lists

Sometimes we want to pass a variable number of arguments into a function. For example:

```java
String.format("%s worked %.2f hours", name, hours);
```

Above is dyadic.

## Verbs and Keywords

Choosing good names for a function can go a long way toward explaining the intent of the function and the order and intent of the arguments.

## Have No Side Effects

- Side effects are lies. Your function promises to do one thing, but it also does other *hidden things*.

- They are devious and damaging mistruths that often result in strange temporal couplings and order dependencies.

- The side effect creates a temporal coupling.

## Output Arguments

- Arguments are most naturally interpreted as *inputs* to a function. If you have been programming for more than a few years, I'm sure you've done a double-take on an argument that was actually an *output *rather than an input.

- In general output arguments should be avoided. If your function must change the state of something, have it change the state of its owning object.

## Command Query Separation

Function should either do something or answer something.

## Prefer Exceptions to Returning Error Codes

- Returning error code from command functions is a subtle violation of command query separation.

- When you return an error code, you create the problem that the caller must deal with the error immediately.
  
  ```java
  if (deletePage(page) == E_OK) {
      if (registry.deleteReference(page.name) == E_OK) {
        if (configKeys.deleteKey(page.name.makeKey()) == E_OK) {
            logger.log("page deleted");
        } else {
            logger.log("configKey not deleted");
        }
    } else {
        logger.log("deleteReference from registry failed");
    }
  } else {
      logger.log("delete failed");
    return E_ERROR;
  }
  ```

- If you use exceptions instead of returned error codes, then the error processing code can be separated from the happy path code and can be simplified.
  
  ```java
  try {
      deletePage(page);
    registry.deleteReference(page.name);
    configKeys.deleteKey(page.name.makeKEY());
  } catch (Exception e) {
      logger.log(e.getMessage());
  }
  ```

## Extract Try/Catch Blocks

It's better to extract the bodies of the `try` and `catch` blocks out into functions of their own.

```java
public void delete(Page page) {
    try {
        deletePageAndAllReferences(page);
    } catch (Exception e) {
        logError(e);
    }
}
```

## Error Handling is One Thing

- Functions should do one thing.

- Error handling is one thing.

- A function that handles errors should do nothing else.

## The Error.java Dependency Magnet

- When the `Error enum` changes, all those other classes need to be recompiled and redeployed, This puts a negative pressure on the `Error` class.

- New exceptions are derivatives of the exception class. They can be added without forcing any recompilation or redeployment.

## Don't Repeat Yourself

- The readability of the whole module is enhanced by the reduction of the duplication.

- Duplication may be the root of all evil in software. Many principles and practices have been created for the purpose of controlling or eliminating it.

## Structure Programming

- Keep your functions small, then occasional multiple `return`, `break` or `continue` statement does not harm and can sometimes even be more expressive than the single-entry, single exit rule.

## How Do You Write Functions Like This?

- Writing software is like any other kind of writing.

- When you write a paper or an article, you get your thoughts down first, then you massage it until it reads well.

- The first draft might be clumsy and disorganized, so you wordsmith it and restructure it and refine it until it reads the way you want it to read.

- Have a suite of unit tests that cover every one of those clumsy lines of code. Then break out whole classes, all the while keeping the tests passing.

## Conclusion

- Functions are the verbs of that language, and classes are the nouns.

- Master programmers think of the systems as stories to be told rather than programs to be written.
