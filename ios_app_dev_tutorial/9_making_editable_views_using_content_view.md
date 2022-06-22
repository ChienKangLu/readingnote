# Making Editable Views - Using Content View

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/using-content-views)

## Section 1 Extract Configuration Methods

As you add more types of collection view cells to the user interface, the cell registration handler becomes unwieldy.

In this section, you’ll isolate the cell configurations by extracting them from the reminder view controller file into an extension in a separate file.

## Section 2 Create a Reusable Layout Function

Using a text field, a date picker, and a text view, the user can modify reminder details when in editing mode.

You’ll pin these subviews to their superviews in all four directions — top, leading, trailing, and bottom — with adjustable padding in each direction.

## Section 3 Create a Custom View with a Text Field

The controls you’ll use for editing in the Today app are all custom subclasses of `UIView`.

To employ the power and styling of UIKit configurations, your custom `UIView`subclasses will conform to the `UIContentView` protocol.

In this section, you’ll create the first of these interactive controls, a custom text field object that displays the title in an editable text area.

## Section 4 Conform to the Content View Protocol

An object that conforms to `UIContentView` requires a configuration property of type `UIContentConfiguration`, which you’ll add in this section.

The configuration that you use in the editable title cell has a `text` property that represents the value included in the text field. In this section, you’ll also create a custom `UIContentConfiguration` type that has a `text`property.

## Section 5 Complete the Content View

Content configurations help keep your user interface in sync with the app’s state. In this section, you’ll ensure that the state of the title text field and the user interface remain synced by updating the user interface whenever the configuration property changes.

Then, you’ll extend `ReminderViewController+CellConfiguration.swift` to include a function that returns a text field configuration, which pairs with your title text field in editing mode.

## Section 6 Display the Content View

In this section, add an editable text field for the reminder’s title to your user interface.

First, add an editable case to the types of rows and an edit title cell item to the editing snapshot.

Then, configure the edit title cell using the configuration that you defined in preceding sections.

Finally, test the editing mode functionality by editing a title in Today.

## Section 7 Create Content Views for the Date and Notes

In the previous two sections, you defined a content view for editing a reminder title in a text field.

In this section, you’ll use the same pattern to complete editing mode for the remaining reminder details. You’ll add content views for editing the reminder date with a date picker and editing the notes with a text view.
