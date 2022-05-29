# Chapter 5: Formatting

When people look under the hood, we want them to be impressed with the nearness, consistency, and attention to detail that perceive.

We want them to perceive that professionals have been at work.

You should take care that your code is nicely formatted.

## The Purpose of Formatting

- Code formatting is *important*.

- Code formatting is about communication, and communication is the professional developer's first order of business.

- The functionality that you create today has a good chance of changing in the next release, but the readability of your code will have a profound effect on all the changes that will ever be made.

## Vertical Formatting

How big should a source file be?

It appears to be possible to build significant systems out of files that are typically 200 lines ling, with an upper limit of 500.

Small files are usually easier to understand than large files are.

### The Newspaper Metaphor

- The name should be simple but explanatory.

- The topmost parts of the source file should provide the high-level concepts and algorithms.

- Detail should increase as we move downward.

### Vertical Openness Between Concepts

- Complete thoughts should be separated from each other with blank lines.

- There are blank lines that separate the package declaration, the import(s), and each of the functions.

### Vertical Density

- If openness separates concepts, then vertical density implies close association.

- Lines of code that are tightly related should appear vertically dense.

### Vertical Distance

- Concepts that are closely related should be kept vertically close to each other.

- We want to avoid forcing out readers to hop around through our source files and classes.

- **Variable Declarations**: Variables should be declared as close to their usage as possible.

- **Instance Variables**: Should be declared at the top of the class.

- **Dependent Functions**: If one function calls another, they should be vertically close, and the caller should be above the callee, if at all possible.
  This makes it easy to find the called functions and greatly enhances the readability of the whole module.

- **Conceptual Affinity**: The stronger that affinity, the less vertical distance there should be between them.
  Affinity might be caused because a group of functions perform a similar operation.

- **Vertical Ordering**: In general we want function call dependencies to point in the downward direction.
  This creates a nice flow down the source code module from high level to low level.
  We expect the low-level details to come last

## Horizontal Form

How wide should a line be?

Programmers clearly prefer short lines. I personally set my limit at 120.

### Horizontal Openness and Density

- We use horizontal white space to associate things that are strongly related and disassociate things that are more weakly related.

- Surround the assignment operators with white space to accentuate them

- Don't put spaces between the function names and the opening parenthesis. This is because the function and its arguments are closely related.

- Separate arguments to accentuate the comma and show that the arguments are separate.

- Use white space to accentuate the precedence of operators. The terms are separated by white space because addition and subtraction are low precedence.

### Horizontal Alignment

- Prefer unaligned declarations and assignments.

- Long length of lists suggests that this class should be split up.

### Indentation

- A source file is a hierarchy rather like an outline.

- Each level of this hierarchy is a scope into which names can be declared and in which declarations and executable statements are interpreted.

- To make this hierarchy of scopes visible, we indent the lines of source code in proportion to their position in the hiearchy.

- **Breaking Indentation**: Whenever I have succumbed to this temptation, I have almost always gone back and put the indentation back in. So I avoid collapsing scopes down to one line.

## Team Rules

A team of developers should agree upon a single formatting style, and then every member of that team should use that style.

We want the software to have a consistent style.
