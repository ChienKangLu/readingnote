# Chapter 1: Clean Code

## There Will Be Code

- We need better programmers.

- We'll be able to tell the difference between good code and bad code.

- We'll know how to transform bad code into good code.

- Some have suggested that we are close to the end of code. Nonsense! We will never be rid of code, because code represents the details of the requirements.
  
  - Specifying requirements that machine can execute  is *programming*.
  
  - Such a specification is *code*.

## Bad Code

- They had rushed the product to market and had made a huge mess in the code. As they added more and more features, the code got worse and worse until they simply could not manage it any longer. *It was the bad code that brought the company down*.

- Of course you have been impeded by bad code, So then - why did you write it?

- *Later equals never*.

## The total Cost of Owning a Mess

- You have probably been slowed down by messy code. The degree of the slowdown can be significant.

- As the mess builds, the productivity of the team continues to decrease, asymptotically approaching zero.

## The Grand Redesign in the Sky

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

## Attitude

- Why does good code rot so quickly into bad code? We are unprofessional.

- We are deeply complicit in the planning of project and share a great deal of the responsibility for any failure.

- It's *your* job to defend the code with equal passion.

- It's unprofessional for programmer to bend to the will of managers who don't understand the risks of making messes.

## The Primal Conundrum

- The *only* way to make the deadline - the only way to go fast - is to keep the code as clean as possible all the time.

## The Art of Clean Code?

- Writing clean code is a lot like painting a picture.

- Being able to recognize clean code from dirty code does not mean that we know how to write clean code.

- **Writing clean code requires the disciplined use of a myriad little techniques applied through a painstakingly acquired sense of "cleanliness". This "code-sense" is the key. Not only does it let us see whether code is good or bad, but it also shows us the strategy for applying our discipline to transform bad code into clean code.**

- A programmer with "code-sense" will look at a messy module and see options and variations.

- The "code-sense" will help that programmer choose the best variation and guide him or her to plot a sequence of behavior preserving transformation to get from here to there.

- A programmer who writes clean code is an artist who can take a blank screen through a series of transformations until it is an elegantly coded system.

## What Is Clean Code?

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

## School of thought

- Don't make the mistake of thinking that we are somehow "right" in any absolute sense. ***There are other schools and other masters that have just as much claim to professionalism as we.***

- The recommendations in this book are things that we have thought long and hard about. We have leaned them through decades of experience and repeated trial and error. ***So whether you agree or disagree, it would be a shame if you did not see and respect, out point of view***.

## We Are Authors

- The next time you write a line of code, remember you are an author, writing for readers who will judge your effort.

- You get the drift. the ratio of time spent reading vs. writing is well over 10:1.

- We are *constantly* reading old code as part of the effort to write new code.

- *Making it easy to read actually makes it easier to write*.

## The Boy Scout Rule

- *Leave the campground cleaner than you found*.

- Isn't continuous improvement an intrinsic part of professionalism? absolutely yes!

## Prequel and Principles

- *Agile Software Development: Principles, Patterns, and Practices (PPP)*

- Single Responsibility Principle (SRP)

- Open Closed Principle (OCP)

- Dependency Inversion Principle (DIP)

## Conclusion

- Books on are don't promise to make you an artist. All it can do is show you the thought processes of good programmers and the tricks, techniques, and tools that they use.
