# Singleton

## Intent

Ensure a class only has one instance, and provide a global point of access to it.

## Motivation

- It's important for some classes to have exactly one instance

- Ensure that no other instance can be created

## Applicability

- There must be exactly one instance of a class, and it must be accessible to clients from a well-known access point

- When the sole instance should be extensible by subclassing, and clients should be able to use an extended instance without modifying their code

## Structure

<img src="./img/singleton_structure.svg" title="" alt="singleton_structure!" data-align="center">

## Collaborations

- Clients access a Singleton instance solely through Singleton's Instance operation.

## Consequences

- Control access to sole instance (how and when clients access it)

- It's easy to configure an application with an instance of this extended class

- The pattern makes it easy to change your mind and allow more than one instance of the Singleton class

## Implementation

Clients access the singleton exclusively through the `Instance` member function. Notice that the constructor is protected.

```c
class Singleton {
public:
    static Singleton* Instance();
protected:
    Singleton();
private:
    static Singleton* _instance;
}
```

Ensure that a Singleton is created and initialized before its first use.

```c
Singleton* Singleton::_instance = 0;

Singleton* Singleton::Instance() {
    if (_instance == 0) {
        _instance = new Singleton;
    }
}
```

Another way to choose the subclass of Singleton by **registry of singletons**:

```c
class Singleton {
public:
    static void Register(const char* name, Singleton*);
    static Singleton* Instance();
protected:
    static Singleton* Lookup(const char* name);
private:
    static Singleton* _instance;
    static List<NameSingletonPair>* _registry;
}
```

```c
Singleton* Singleton::Instance() {
    if (_instance == 0) {
        const char* singletonName = getenv("SINGLETON");
        // user or environment supplies this at startup
        
        _instance = Lookup(singletonName);
        // Lookup returns 0 if there's no such singleton
    }
    return _instnace;
}
```

```c
MySingleton::MySingleton() {
    // ...
    Singleton::Register("MySingleton", this);
}
```


