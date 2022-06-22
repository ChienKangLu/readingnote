# Adding, Deleting Reminders

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/adding-and-deleting-reminders)

## Section 1 Create an Add Action

In editing mode, `ReminderViewController` displays controls that allow users to modify the details of an existing reminder. You’ll use the same view controller and controls to enable users to create new reminders.

You’ll start by adding an `isAddingNewReminder` property to `ReminderViewController`. Then, you’ll create a method that displays a `ReminderViewController` in a modal view, which covers the list view when a user creates a new reminder. And you’ll embed the `ReminderViewController` in a navigation controller so that you can add Cancel and Done buttons to the user interface.

## Section 2 Connect the Target-Action Pair

You’ll add a button to the navigation bar and connect it to the `didPressAddButton(_ :)` method that you created in the last section.

## Section 3 Add a New Reminder to the Model

In this section, you’ll complete the add reminder functionality. You’ll update the the reminder view controller’s `onChange`closure to save the reminder to the model when a user taps the Done button.

## Section 4 Delete a Reminder

In this section, you’ll add swipe-to-delete functionality to the reminder list.
