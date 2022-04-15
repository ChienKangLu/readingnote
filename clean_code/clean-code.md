# Clean Code

*Writing clean code is what you must do in order to call yourself a professional.*

*There is no reasonable excuse for doing anything less than your best.*

## Forward

- Honesty in small things is not a small thing.

- Small things matter.

- This is a book about humble concerns whose value is nonetheless far from small.

- God is in the details.

- Software architecture has an important place in development.

- Responsible professionals give some time to thinking and planning at the outset of a project.

- Attentiveness to detail is an even more critical foundation of professionalism than is any grand vision. First, it is through practice in the small that professionals gain proficiency and trust for practice in the large. Second, the smallest bit of sloppy construction completely dispels the charm of the larger while. That is what clean code about.

- In software, 80% or more of what we do is quaintly called "maintenance": the act of repair.

- 5S is the pillars of Total Productive Maintenance (TPM), which is the foundation of Lean:
  
  - Organization. Suitable naming is crucial
  
  - Tidiness. *A place for everything, and everything in its place*. A piece of code should be where you expect to find it. If not, you should re-factor to get it there.
  
  - Cleaning. Get rid of the comments and comment-out code lines that capture history or wishes for the future.
  
  - Standardization. The group agrees about how to keep the workplace clean
  
  - Self-discipline. This means having the discipline to follow the practices and to frequently reflect on one's work and be willing to change.

- Inspect the machine every day and fix wearing parts before they break.

- Build machines that are more maintainable in the first place.

- Making your code readable is as important as making it executable.

- We should re-do major software chunks from scratch weekly, daily, hourly or so to sweep away creeping cruft (unwanted code).

- *Neatness as a remedy for every degree of evil.*

- As beautiful as a house is, a messy desk robs it of its splendor.

- *He who is faithful in little is faithful in much.*

- Being eager to re-factor at the responsible time, strengthening one's position for subsequent big decisions, rather than putting it off:
  
  - *A stitch in time saves nine.*
  
  - *Don't putt off until tomorrow what you can do today.*

- Calibrating the place of small, individual efforts in a ground whole:
  
  - *Mighty oaks from little acorns grow.*

- Integrating simple preventive work into everyday life:
  
  - *An ounce of prevention is worth a proud of cure.*
  
  - *An apple a day keeps the doctor away.*

- Our culture should be with attentiveness to detail.

- You should name a variable using the same care with which you name a first-born child.

- As every homeowner knows, such care and ongoing refinement never come to an end.

- Design is ever ongoing not only as we add a new room to a house, but as we are attentive to repainting, replacing worn carpet, or upgrading the kitchen sink.

- A poem is never done and bears continual rework, and to stop working on it is abandonment. Such preoccupation with detail is common to all endeavors of excellence.

- We abandon our code early, not because it is done, but because our value system focuses more on outward appearance than on the substance of what we deliver.

- The inattentiveness costs us in the end: *A bad penny always shows up*.

- Research, neither in industry nor in academia, humbles itself to the lowly station of keeping code clean.

- Consistent indentation style was one of the most statistically significant indicators of low bug density.

- The Japanese worldview understands the crucial value of the everyday worker and, more so, of the systems of development that owe to the simple, everyday actions of those workers.

- Quality is the result of a million selfless acts of care.

- "The code is the design" and "Simple code" are their mantra.

- It is crucial to continuously adopt the humble stance that the design lives in the code.

- We should view our code as the beautiful articulation of noble efforts of design-design as a process, not a static endpoint.

- Abstraction is evil. Code is anti-evil, and clean code is perhaps divine.

- Not just to pay attention to small things, but also to be honest in small things. This means being honest to the code,  honest to our colleagues about the state of our code and, most of all, being honest with ourselves about our code.

- The concerns that lie squarely in the center of Agile values:
  
  - Did we Do our Best to "leave the campground cleaner than we found it"?
  
  - Did we re-factor our code before checking in?

- **Neither architecture nor clean code insist on perfection, only on honesty and doing the best we can. *To err is human; to forgive, divine*. In Scrum, we make everything visible. We air our dirty laundry. We are honest about the state of our code because code is never perfect.**

## Introduction

- There are two parts of learning craftsmanship: knowledge and work.
  
  - You must gain the knowledge of principles, patterns, and heuristics that a craftsman knows.
  
  - You must also grind that knowledge into your fingers, eyes, and gut by working hard and practicing.

- Learning to write clean code is *hard work*. It requires  more than just knowledge of principles and patterns. You must *sweat* over it.
  
  - You must practice it yourself, and watch yourself failed.
  
  - You must watch other practice it and fail. You must see them stumble and retrace their steps. You must see them agonize over decisions and see the price they pay for making those decisions the wrong way.

- Think along the same paths that we thought, then you will gain a much richer understanding of those principles, patterns and heuristics.

- They'll have become part of you in the same way that a bicycle becomes and extension of your will when you have mastered how to ride it.

## Chapter 1: Clean Code

### There Will Be Code

- We need better programmers.

- We'll be able to tell the difference between good code and bad code.

- We'll know how to transform bad code into good code.

- Some have suggested that we are close to the end of code. Nonsense! We will never be rid of code, because code represents the details of the requirements.
  
  - Specifying requirements that machine can execute  is *programming*.
  
  - Such a specification is *code*.

### Bad Code

- They had rushed the product to market and had made a huge mess in the code. As they added more and more features, the code got worse and worse until they simply could not manage it any longer. *It was the bad code that brought the company down*.

- Of course you have been impeded by bad code, So then - why did you write it?

- *Later equals never*.

### The total Cost of Owning a Mess

- You have probably been slowed down by messy code. The degree of the slowdown can be significant.

- As the mess builds, the productivity of the team continues to decrease, asymptotically approaching zero.

### The Grand Redesign in the Sky

Spending time keeping your code clean is not just cost effective; it's a matter of professional survival.

```bash
                                            --------------------------------
                                            |                              |
                                            v                              |
+------------------+      new team    +----------+               +-------------------------+
| Odious code base | ---------------> | redesign | --10 years--> | new system is such mess |
+------------------+   |              +----------+               +-------------------------+
                       |
                       |  old team    +---------------------+
                       -------------> | maintain old system |
                                      +---------------------+
```

### Attitude

- Why does good code rot so quickly into bad code? We are unprofessional.

- We are deeply complicit in the planning of project and share a great deal of the responsibility for any failure.

- It's *your* job to defend the code with equal passion.

- It's unprofessional for programmer to bend to the will of managers who don't understand the risks of making messes.

### The Primal Conundrum

- The *only* way to make the deadline - the only way to go fast - is to keep the code as clean as possible all the time.

### The Art of Clean Code?

- Writing clean code is a lot like painting a picture.

- Being able to recognize clean code from dirty code does not mean that we know how to write clean code.

- **Writing clean code requires the disciplined use of a myriad little techniques applied through a painstakingly acquired sense of "cleanliness". This "code-sense" is the key. Not only does it let us see whether code is good or bad, but it also shows us the strategy for applying our discipline to transform bad code into clean code.**

- A programmer with "code-sense" will look at a messy module and see options and variations.

- The "code-sense" will help that programmer choose the best variation and guide him or her to plot a sequence of behavior preserving transformation to get from here to there.

- A programmer who writes clean code is an artist who can take a blank screen through a series of transformations until it is an elegantly coded system.

### What Is Clean Code?

- Elegant and efficient code
  
  - ***Straightforward logic***: hard for bugs to hide.
    
    - Reading it should make you smile the way a well-crafted music box or well-designed car would.
  
  - ***Minimal dependencies***: ease maintenance.
  
  - ***Complete error handling***
    
    - Error handling should be complete.
    
    - Pay attention to details.
    
    - Clean code exhibits close attention to detail.
  
  - ***Close to optimal performance***: not tempt people to make the code messy with unprincipled optimizations.
    
    - Bad code tempts the mess to grow.
    
    - A building with broken windows looks like nobody cares about it.
  
  - ***Clean code does one thing well***
    
    - Bad code tries to do too much, it has muddled intent and ambiguity of purpose. Clean code is *focused*.
    
    - Each function, each class, each module exposes a single-minded attitude that remains entirely undistributed, and unpolluted, by the surrounding details.

- Clean code is simple and direct
  
  - ***Reads like well-written prose***
    
    - Readability perspective
  
  - ***Clean code never obscures the designer's intent***
    
    - Clean code should clearly expose the tensions in the problem to be solved.
  
  - ***Full of crisp abstraction and straightforward lines of control***
    
    - One code should be matter-of -fact as opposed to speculative.
    
    - Our readers should perceive us to have been decisive.

- ***Clean code can be read, and enhanced by a developer other than its original author***
  
  - Clean code makes it easy for other developer to enhance it.
  
  - Code that is easy to read
  
  - Code that is easy to change

- ***It has unit and acceptance tests***
  
  - Test Driven Development
  
  - Code, without tests, is not clean.

- ***It has meaningful meaningful names***

- ***It provides one way rather than many ways for doing one thing***

- ***It has minimal dependencies, which are explicitly defined, and provides a clear and minimal API***
  
  - Smaller is better.

- ***Code should be literate***
  
  - The code should be composed in such a form as to make it readable by humans.

- ***Clean code always looks like it was written by someone who cares*. There is nothing you can do to make it better**

- ***Simple code***
  
  - ***Runs all tests***
  
  - ***Contains no duplication*** (Reduced duplication)
    
    - When the same thing is done over and over, it's a sign that there is an idea in out mind that is not well represented in the code.
  
  - ***Expresses all the design ideas that are in the system*** (High expressiveness)
    
    - Meaningful names
    
    - Whether an object or method is doing more than one thing
      
      - Object: it needs to be broken into two or more objects.
      
      - Method: use the Extract Method refactoring on it, resulting in one method that says more clearly what it does.
  
  - ***Early building of simple abstractions***

- ***You know you are working on clean code when each routine you read turns out to be pretty much what you expected***
  
  - When you read clean code you won't be surprised at all.
  
  - Programs that are *that* clean are so profoundly well written that you don't even notice it.
  
  - Beautiful code make the language look like it was made for the problem! So it's our responsibility to make the language look simple!
  
  - It is not the language that makes programs appear simple. It is the programmer that make the language appear simple.

### School of thought

- Don't make the mistake of thinking that we are somehow "right" in any absolute sense. ***There are other schools and other masters that have just as much claim to professionalism as we.***

- The recommendations in this book are things that we have thought long and hard about. We have leaned them through decades of experience and repeated trial and error. ***So whether you agree or disagree, it would be a shame if you did not see and respect, out point of view***.

### We Are Authors

- The next time you write a line of code, remember you are an author, writing for readers who will judge your effort.

- You get the drift. the ratio of time spent reading vs. writing is well over 10:1.

- We are *constantly* reading old code as part of the effort to write new code.

- *Making it easy to read actually makes it easier to write*.

### The Boy Scout Rule

- *Leave the campground cleaner than you found*.

- Isn't continuous improvement an intrinsic part of professionalism? absolutely yes!

### Prequel and Principles

- *Agile Software Development: Principles, Patterns, and Practices (PPP)*

- Single Responsibility Principle (SRP)

- Open Closed Principle (OCP)

- Dependency Inversion Principle (DIP)

### Conclusion

- Books on are don't promise to make you an artist. All it can do is show you the thought processes of good programmers and the tricks, techniques, and tools that they use.

## Chapter 2: Meaningful Names

### Use Intention-Revealing Names

- The name of a variable should answer all the big questions:
  
  - why it exists
  
  - what is does
  
  - how it is used

- If a name requires a comment, then the name does not reveal its intent.

#### Example 1

The name `d` reveals nothing:

```java
int d; // elapsed time in days
```

We should choose a name that specifics what is being measured and the unit of that measurement:

```java
int elapsedTimeInDays;
int daysSinceCreation:
int daysSinceModification;
int fileAgeInDays;
```

#### Example 2

What is the purpose of this code?

```java
public List<int[]> getThem() {
    List<int[]> list1 = new ArrayList<int[]>();
    for (int[] x : theList)
        if (x[0] == 4)
            list1.add(x);
    return list1;
}
```

Just by giving these concepts names we can improve the considerably.

```java
public List<int[]> getFlaggedCells() {
    List<int[]> flaggedCells = new ArrayList<int[]>();
    for (int[] cell : gameBoard)
        if (cell[STATUS_VALUE] == FLAGGED)
            flaggedCells.add(cell);
    return flaggedCells;
}
```

We can go further and write a simple class for cells instead of using an array of `ints`. It can include an intention-revealing function (call it `isFlagged`) to hide the magic numbers.

```java
public List<Cell> getFlaggedCells() {
    List<Cell> flaggedCells = new ArrayList<Cell>();
    for (Cell cell : gameBoard)
        if (cell.isFlagged())
            flaggedCells.add(cell);
    return flaggedCells;
}
```

### Avoid DisInformation

- Avoid words whose entrenched meanings vary from our intended meaning. For example, `hp`, `aix` and `sco` would be poor variable names because they are the names of Unix platforms or variants.

- Do not refer to a grouping of accounts as an `accountList`unless it's actually a `List`. So `accountGroup` or `bunchOfAccounts` or just plain `accounts` would be better. 
  
  - Even if the container is a `List`, it's probably better not to encode the container type into the name.

- Beware of using names which vary in small ways.

- Spelling similar concepts similarly is *information*. Using inconsistent spelling is *disinformation*.
  
  - Lower-case `L` and uppercase `O` look almost entirely like the constants one and zero, respectively.

### Make Meaningful Distinctions

- Number-series naming `(a1, a2, .. aN)` is opposite of intentional naming
  
  ```java
  public static void copyChars(char a1[], char a2[]) {
      for (int i = 0; i < a1.length; i++) {
          a2[i] = a1[i];
      }
  }
  ```
  
  The function reads much better when `source` and `destination` are used for argument names.
  
  ```java
  public static void copyChars(char source[], char destination[]) {
      for (int i = 0; i < source.length; i++) {
          destination[i] = source[i];
      }
  }
  ```

- Noise words are meaningless distinction
  `Product`, `ProductInfo` and `ProductData` mean nothing different. `Info` and `Data` are indistinct noise words like `a`, `an` and `the`.

- Noise words are redundant
  
  - `variable` in variable name is redundant
  
  - `table` in table name is redundant
  
  - `NameString` is redundant for `Name` when it's never a floating point number

- Distinguish names in such a that the reader knows what the differences offer
  
  - `getActiveAccount()`, `getActiveAccounts()` and `getActiveAccountInfo()` are indistinguishable
  
  - `moneyAmount` is indistinguishable from `money`
  
  - `customerInfo` is indistinguishable from `customer`
  
  - `accountData` is indistinguishable from `account`
  
  - `theMessage` is indistinguishable from `message`

### Use Pronounceable Names

- If you can not pronounce it, you can't discuss it without sounding like an idiot. This matters because programming is a social activity.
  
  ```java
  class DtaRcrd102 {
      private Date genymdhms;
      private Date modymdhms;
      private final String pszqint = "102";
  }
  ```
  
  to
  
  ```java
  class Customer{
      private Date generationTimestamp;
      private Date modificationTimestamp;
      private final String recordId = "102";
  }
  ```

### Use Searchable Names

- Single-letter names and numeric constants have a particular problem in that they are not easy to locate across a body of text.

- Long names trumps shorter names.

- Searchable name trumps a constant in code.

- Single-letter names can ONLY be used as local variables inside short methods. *The length of a name should correspond to the size of tits scope*.

- If a variable or constant might be seen or used in multiple places in a body of code, it is imperative to give it a search-friendly name.
  
  ```
    for (int j=0; j<34; j++) {
      s += (t[j]*4)/5;
  }
  ```
  
  to
  
  ```
  int realDaysPerIdealDay = 4;
  const int WORK_DAYS_PER_WEEK = 5;
  int sum = 0;
  for (int j = 0; j < NUMBER_OF_TASKS; j++) {
      int realTaskDays = taskEstimate[j] * realDaysPerIdealDay ;
      int realTaskWeeks = (realTaskDays / WORK_DAYS_PER_WEEK);
      sum += realTaskWeeks ;
  }
  ```

### Avoid Encodings

- Encoding type or scope information into names simply adds an extra burden of deciphering.

- Encoded names are seldom pronounceable and are easy to mis-type.

- No need to prefix member variables with `m_` anymore.

- Prefer to leave interfaces unadorned. So if I must encode either the interface or the implementation, I choose the implementation.

### Avoid Mental Mapping

- Reader shouldn't have to mentally translate your names into other names they already know.

- This problem generally arises from a choice to use neither problem domain terms nor solution domain terms.

- One difference between a smart programmer and a professional programmer is that the professional understands that *clarity is king*. Professionals use their powers for good and write code that others can understand.

### Class Names

- Classes and objects should have noun or noun phrase names.

- A class name should not be a verb.

### Method Names

- Methods should have verb or verb phrase names like `postPayment`, `deletePage` or `save`.

- Accessors, mutators and predicates should be named for their value and prefixed with `get`, `set`and `is` according to javabean standard.
  
  ```java
  string name = employee.getName();
  customer.setName("mike");
  if (paycheck.isPosted())...
  ```

- When constructors are overloaded, use static factory methods with names that describe the arguments.
  
  ```java
  Complex fulcrumPoint = new Complex(23.0);
  // Making the corresponding constructors private.
  Complex fulcrumPoint = Complex.FromRealNumber(23.0); // better
  ```

### Don't Be Cute

- Choose clarity over entertainment value.

- Say what you mean. Mean what you say.

### Pick One Word per Concept

- Pick one word for one abstract concept and stick with it.

- The function names have to stand alone, and they have to be consistent in order for you to pick the corrected method without any additional exploration.

- A consistent lexicon is a great boon to the programmers who must use your code.

### Don't Pun

- Avoid using the same word for two purposes.

### Use Solution Domain Names

- Use computer science (CS) terms, algorithm names, patter names math terms, and so forth.

- It is not wise to draw every name from the problem domain.

### Use Problem Domain Names

- When there is no "programmer-eese" for what you're doing, use the name from the problem domain.

- Separating solution and problem domain is part of the job of good programmer and designer.
