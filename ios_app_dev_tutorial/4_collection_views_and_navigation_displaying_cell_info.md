# Collection Views and Navigation — Displaying Cell Info

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/displaying-cell-info)

## Section 1 Format the Date and Time

### Time

```swift
extension Date {
    ...
    let timeText = formatted(date: .omitted, time: .shortened)
    ...
    let dateText = formatted(.dateTime.month(.abbreviated).day())
    ...
    formatted(.dateTime.month().day().weekday(.wide))
}
```

### Localize String

```swift
NSLocalizedString("Today at %@", comment: "Today at time format string")
String(format: timeFormat, timeText)
```

## Section 2 Organize View Controllers

Because they have many responsibilities in UIKit apps, view controller files can be large. Reorganizing the view controller responsibilities into separate files and extensions makes it easier to find errors and add new features later.

## Section 3 Change the Cell Background Color

```swift
var backgroundConfiguration = UIBackgroundConfiguration.listGroupedCell()
// backgroundConfiguration.backgroundColor = .todayListCellBackground
backgroundConfiguration.backgroundColor = UIColor(red: 1.0, green: 0.8, blue: 0.9, alpha: 1.0)
```

## Section 4 Display the Reminder Complete Status

Create Image with SF symbol.

```swift
let symbolName = reminder.isComplete ? "circle.fill" : "circle"
let symbolConfiguration = UIImage.SymbolConfiguration(textStyle: .title1)
let image = UIImage(systemName: symbolName, withConfiguration: symbolConfiguration)
```

Create Button.

```swift
let button = UIButton()
button.setImage(image, for: .normal)
```
