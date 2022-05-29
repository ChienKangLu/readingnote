# Chapter 4: Comments

If our programming languages were expressive enough, or if we had talent to subtly wield those languages to express our intent, we would not need comments very much - perhaps not at all.

The proper use of comments is to compensate for our failure to express ourself in code.

So when you find yourself in a position where you need to write a comment, think it through and see whether there isn't some way to turn the tables and express yourself in code.

Inaccurate comments are far worse than no comments at all.

***<u>Truth can only be found in one place: the code.</u>***

## Comments Do Not Make Up for Bad Code

- Clear and expressive code with few comments is far superior to cluttered and complex code with lots of comments.

- Rather than spend your time writing the comments that explain the mess you've made, spend it cleaning that mess.

## Explain Yourself in Code

In many cases it's simply a matter of creating a function that says the same thing as the comment you want to write

```java
// Check to see if the employee is eligible for full benefits
if ((employee.flags & HOURLY_FLAG) && (employee.age > 65))
```

or this?

```java
if (employee.isEligibleForFullBenefits())
```

## Good Comments

### Legal Comments

Sometimes our corporate coding standard force us to write certain comments for legal reasons.

### Informative Comments

It's sometimes useful to provide basic information with a comment.

```java
// format matched kk:mm:ss EEE, MM dd, yyy
Pattern timeMatch Pattern.compile("\\d*\\d*\\d* \\w*, \\w* \\d*, \\d*");
```

### Explanation of Intent

Sometimes a comments goes beyond just useful information about the implementation and provides the intent behind decision.

### Clarification

Sometimes it is just helpful to translate the meaning of some obscure argument or return value into something that's readable.

```java
assertTrue(a.compareTo(a) == 0); // a == a
assertTrue(a.compareTo(b)  != 0); // a != b
```

### Warning of consequences

Sometimes it is useful to warn other programmers about certain consequences.

### TODO Comments

`TODO`s are jobs that the programmer thinks should be done.

### Amplification

A comment may be used to amplify the importance of something that may otherwise seem inconsequential.

### Javadocs in Public APIs

There is nothing quite so helpful and satisfying as a well-described public API. If you are writing a public API, then you should certainly write good javadocs for it.

## Bad Comments

### Mumbling

- If you decide to write a comment, then spend the time necessary to make sure it is the best comment you can write.

- Any comment that forces you to look in another module for the meaning of that comment has failed to communicate to you and is not worth the bits it consumes.

### Redundant Comments

- The comment probably takes longer to read than the code itself.

- It's certainly not more informative than the code.

- It does not justify the code, or provide intent or rationale.

- It's not easier to read than the code.

### Misleading Comments

- Sometimes, with all the best intentions, a programmer makes a statement in his comments that isn't precise enough to be accurate.

- The subtle bit of misinformation, couched in a comment that is harder to read than the body of the code.

### Mandated Comments

It is just plain silly to have a rule that says that every function must have a javadoc, or every variable must have a comment.

### Journal Comments

- Long ago there was a good reason to create and maintain these log entries at the start of every module.

- Nowadays, we have source code control.

### Noise Comments

- Sometimes you see comments that are nothing but noise. They restate the obvious and provide no information.

- Rather than venting in a worthless and noisy comment, the programmer should have recognized that his frustration could be resolved by improving the structure of his code.

- Replace the temptation to create noise with the determination to clean your code. You;ll find it makes you a better and happier programmer.

### Scary Noise

- Javadocs can also be noisy

### Don't Use a Comment When You Can Use a Function or Variable

```java
// does the module from the global list <mod> depend on the
// subsystem we are part of?
if (smodule.getDependSubsystems().contains(subSysMod.getSubSystem()))
```

This could be rephrased without the comments as

```java
ArrayList moduleDependees = smodule.getDependSubsystems();
String ourSubSystem = subSysMod.getSubSystem();
if (moduleDependees.contains(ourSubSystem ))
```

### Position markers

```java
// Actions ///////////////////////////////
```

### Closing Brace Comments

If you find yourself wanting to mark your closing braces, try to shorten your functions instead.

### Attributes and Bylines

The source code control system is a better place for this kind of information.

### Commented-Out Code

Source code control system will remember the code for us. We don't have to comment it out any more. Just delete the code. We don't lose it. Promise.

### HTML Comments

It makes the comments hard to read in one place where they should be easy to read - the editor/IDE.

### Nonlocal Information

If you write a comment, then make sure it describes the code it appears near. Don't offer systemwide information in the context of a local comment.

### Too much information

Don't put interesting historical discussions or irrelevant descriptions of details into your comments.

### Inobvious Connection

- The connection between a comment and he code it describes should be obvious.

- If you are going trouble to write a comment, then at least you'd like the reader to be able to look at the comment and the code and understand what the comment is talking about.

### Function Headers

- Short functions don't need much description.

- A well-chosen name for a small function that does one thing is usually better than a comment header.

### Javadocs in Nonpublic Code

As useful as javadocs are for public APIs.
