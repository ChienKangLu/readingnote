# Chapter 6: Objects and Data Structures

- Keep our variables private.

- Keep the freedom to change their type or implementation on whim or an impulse.

## Data Abstraction

- Hiding implementation is not just a matter of putting a layer of functions between the variables.

- Hiding implementation is about abstractions. Rather it exposes abstract interfaces that allow its users to manipulate the essence of the data, without having to know its implementation.

- We do not want to expose the details of our data. Rather we want to express our data in abstract terms.

## Data/Object Anti-Symmetry

- **Object**s hide their data behind abstractions and expose functions that operate on the data.

- **Data structure** expose their data and have no meaningful functions.

- *Procedural code makes it east to add new functions without changing the existing data structures. OO code, on the other hand, makes it easy to add new classes without changing functions.*

- *Procedural code makes it hard to add new data structures because all the functions must change. OO code makes it hard to add new functions because all classes must change.*

- The things that are hard for OO are easy for procedures, and the things that are hard to procedures are easy to OO!

## The Law of Demeter

- The Law of Demeter that says a module should not know about the innards of the *object*s it manipulates.

- An objects should not expose its internal structure through successors because to do so is to expose, rather than to hide. its internal structure.

## Train Wrecks

Chain of calls like this are generally considered to be sloppy style and should be avoided.

```java
final String outputDir = ctxt.getOptions().getScratchDir().getAbsolutePath();
```

It is usually best split them up as follows:

```java
Options opts = ctxt.getoptions();
File scratchDir = opts.getScratchDir();
final String outputDir = scratchDir.getAbsolutePath();
```

## Hybrids

- This confusion sometimes leads to unfortunate hybrid structures that are half object and half data structure.

- Such hybrids make it hard to add new functions but also make it hard to add new data structures. They are the worst of both worlds. Avoid creating them.

## Hiding Structure

Neither option feels good:

```java
ctxt.getAbsolutePathOfScratchDirectoryOption();
// or
ctx.getScratchDirectoryOption().getAbsolutePath();
```

If `ctxt` is and object, we should be telling it to *do something*; we should not be asking it about its internals.

The following is something that seems like an object should do:

```java
BufferedOutputStream bos = ctxt.createScratchFileStream(c;assFileNames);
```

## Data Transfer Objects

- The quintessential form of a data structure is a class with public variables and no functions. This is sometimes called a data transfer object, or DTO.

- Somewhat more common is the "bean" form, which have private variables manipulated by getters and setters.

## Active Record

- A special forms of DTOs.

- They are data structures with public  variables; but they typically have navigational method like `save` and `find`

- Typically these Active Records are direct translations from database tables, or other data sources.

- By putting business rule methods in Active Record is awkward because it creates a hybrid tween a data structure and an object.

- Treat Active Record as a data structure and to create separate objects that contain the business rules and that hide their internal data.

## Conclusion

- **Object**s expose behavior and hide data.
  
  - Pro: This makes it easy to add new kind of objects without changing existing behaviors.
  
  - Con: Makes it hard to add new behaviors to existing objects.

- **Data structure**s expose data and have no significant behavior.
  
  - Pro: Makes it easy to add new behaviors to existing data structures.
  
  - Con: Makes it hard to add new data structures to existing functions.


