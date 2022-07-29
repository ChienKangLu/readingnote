# Error Handling

- Things can go wrong, we as programmers are responsible for making sure that our code does what it needs to do

- Error handling is important, but if it obscures logic, it's wrong

## Use exceptions Rather Than Return Codes

- Return code clutters the caller

- It is better to throw an exception when you encounter an error

- The code is better because two concerns that were tangled, the algorithm for device shutdown and error handling, are now separated.

- You can look at each of those concerns and understand then independently

## Write Your Try-Catch-Finally Statement First

- Exemptions define a a scope

- Try to write tests that force exceptions, and then add behavior to your handler to satisfy your tests

## Use Unchecked Exceptions

- We have to decide - really - whether checked exceptions are worth their price

- The price of checked exceptions is an Open/Closed Principle violation.

- This means that a change at a low level of software can force signature changes on many higher levels

- Encapsulation is broken because all functions in the path of a throw must know about details of that low-level exception

- Checked exceptions can sometimes be useful if you are writing a critical library: You must catch them. But in general application envelopment the dependency costs outweigh the benefits

## Provide Context with Exception

- Each exception that you throw should provide enough context to determine the source and location of an error

- Create informative error messages

- Mention the operation that failed and the type of failure.

## Define Exception Classes in Terms of a Caller's Needs

- Classify errors

- Our most important concern should be *how they are caught*

- Wrapping API that we are calling and making sure that it returns a common exception type

- In fact, wrapping third-party APIs is a best practice. When you wrap a third-party API, you minimize your dependencies upon it

- Wrapping also makes it easier to mock out third-party calls when you are testing your own code

## Define the Normal Flow

- SPECIAL CASE PATTERN

- Creates a class or configuration an object so that it can handles a special case for you

## Don't Return Null

- When we return `null`, we are essentially creating work for ourselves and foisting problem upon our callers.

- If you are tempted to return `null` from a method, consider throwing an exception or returning a SPECIAL CASE object instead.

## Don't Pass Null

- Returning `null` from methods is bad, but passing `null` into methods is worse

## Conclusion

Clean code is readable, but it is also be robust.

We can write robust clean code if we see error handling as a separate concern, something that is view-able independently of our main logic


