# Chapter 2: Meaningful Names

## Use Intention-Revealing Names

- The name of a variable should answer all the big questions:
  
  - why it exists
  
  - what is does
  
  - how it is used

- If a name requires a comment, then the name does not reveal its intent.

### Example 1

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

### Example 2

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

## Avoid DisInformation

- Avoid words whose entrenched meanings vary from our intended meaning. For example, `hp`, `aix` and `sco` would be poor variable names because they are the names of Unix platforms or variants.

- Do not refer to a grouping of accounts as an `accountList`unless it's actually a `List`. So `accountGroup` or `bunchOfAccounts` or just plain `accounts` would be better. 
  
  - Even if the container is a `List`, it's probably better not to encode the container type into the name.

- Beware of using names which vary in small ways.

- Spelling similar concepts similarly is *information*. Using inconsistent spelling is *disinformation*.
  
  - Lower-case `L` and uppercase `O` look almost entirely like the constants one and zero, respectively.

## Make Meaningful Distinctions

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

## Use Searchable Names

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

## Avoid Encodings

- Encoding type or scope information into names simply adds an extra burden of deciphering.

- Encoded names are seldom pronounceable and are easy to mis-type.

- No need to prefix member variables with `m_` anymore.

- Prefer to leave interfaces unadorned. So if I must encode either the interface or the implementation, I choose the implementation.

## Avoid Mental Mapping

- Reader shouldn't have to mentally translate your names into other names they already know.

- This problem generally arises from a choice to use neither problem domain terms nor solution domain terms.

- One difference between a smart programmer and a professional programmer is that the professional understands that *clarity is king*. Professionals use their powers for good and write code that others can understand.

## Class Names

- Classes and objects should have noun or noun phrase names.

- A class name should not be a verb.

## Method Names

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

## Don't Be Cute

- Choose clarity over entertainment value.

- Say what you mean. Mean what you say.

## Pick One Word per Concept

- Pick one word for one abstract concept and stick with it.

- The function names have to stand alone, and they have to be consistent in order for you to pick the corrected method without any additional exploration.

- A consistent lexicon is a great boon to the programmers who must use your code.

## Don't Pun

- Avoid using the same word for two purposes.

## Use Solution Domain Names

- Use computer science (CS) terms, algorithm names, patter names math terms, and so forth.

- It is not wise to draw every name from the problem domain.

## Use Problem Domain Names

- When there is no "programmer-eese" for what you're doing, use the name from the problem domain.

- Separating solution and problem domain is part of the job of good programmer and designer.

## Add Meaningful Context

- You need yo place names in context for your reader by enclosing them in well-named classes, functions, or namespaces.

## Don't Add Gratuitous Context

- Shorter names are generally better than longer ones, so long as they are clear. And no more context to a name than is necessary.

## Final Words

- The hardest thing about choosing food names is that it requires good descriptive skills and a shared cultural background.

- You will probably end up surprising someone when you rename, just like you might with any other code improvement.
