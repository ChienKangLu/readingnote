# Making Editable Views - Managing Content Views

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/managing-content-views)

## Introduction

Content views let you focus on information that you want to display in a view, without worrying about when to update or how to style a view. They work with content configurations to keep your app’s information and user interface in sync.

## Creating a Custom Content View

```swift
class PriorityContentView : UIView, UIContentView {

    // define the user interface elements
    let priorityLabel = UILabel()
    let prioritySlider = UISlider()
    var priorityStack = UIStackView()

    init() {
        // apply style to the user interface
        priorityStack = UIStackView(arrangedSubviews: [priorityLabel, prioritySlider])
        priorityStack.axis = .vertical
        self.addSubview(priorityStack)

        priorityLabel.textAlignment = .center
        priorityLabel.textColor = .white

        prioritySlider.maximumValue = 10.0
        prioritySlider.minimumValue = 0.0
        prioritySlider.addTarget(self, action: #selector(self.sliderValueDidChange(_:)), for: .valueChanged)

        // layout stack in superview
        ...
    }
}
```

## Creating a Custom Content Configuration

```swift
struct ReminderContentConfiguration: UIContentConfiguration {
    var reminder: Reminder // a copy of the model

    func makeContentView() -> UIView & UIContentView {
        return PriorityContentView(self)
    }
    func updated(for state: UIConfigurationState) -> ReminderContentConfiguration {
        return self
    }
    mutating func updatePriority(to newPriority: Int) {
        reminder.currentPriority = newPriority
    }
}
```

## Configuring Your Custom Content View

```swift
var configuration: UIContentConfiguration {
    didSet {
        self.configure(configuration: configuration)
    }
}
```

The `didSet` observer reconfigures the view every time the configuration — including the reminder that it contains — changes, ensuring that the view always represents the updated model data.

The content view accepts an external configuration in the initializer and uses it to configure the view.

```swift
init(_ configuration: UIContentConfiguration) {
    self.configuration = configuration

    // apply style and lay out the user interface
    ...

    self.configure(configuration: configuration) // to be added
}
```

## Applying the Custom Content Configuration

```swift
func configure(configuration: UIContentConfiguration) {
    guard let configuration = configuration as? ReminderContentConfiguration else { return }
    self.priorityLabel.text = configuration.reminder.title + " (priority: \(configuration.reminder.currentPriority))"
    self.prioritySlider.value = Float(configuration.reminder.currentPriority)
}
```

Finally, you’ll also need to apply the configuration when the slider value changes:

```swift

```
