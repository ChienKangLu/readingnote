# Making Editable Views - Editing reminders

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/editing-reminders)

## Section 1 Add a Working Reminder

In this section, you’ll create a temporary reminder to store any edits that the user makes. You’ll apply those changes to the permanent reminder when the user exits editing mode.

## Section 2 Make the Text Configuration Editable

In this section, you’ll set up the text field to edit the reminder title. You’ll define a method that executes every time the text in the text field changes. And you’ll ensure that the detail view updates with the latest changes when the user leaves editing mode.

## Section 3 Make the Date Configuration Editable

In this section, you’ll set up the date picker to edit the reminder date and time. You’ll define a method that executes every time the date in the date picker changes. And you’ll ensure that the detail view displays the updates when the user leaves editing mode and returns to view mode.

## Section 4 Make the Notes Configuration Editable

In this section, you’ll set up the text view to edit the reminder notes. You’ll define a method that executes every time the notes in the text view change. And you’ll ensure that the detail view displays the latest changes when the user leaves editing mode and returns to view mode.

## Section 5 Cancel Edits

In this section, you’ll add a Cancel button to allow the user to leave editing mode without saving changes. You’ll also restore the working reminder to its original state so the user can choose to edit the reminder again.

## Section 6 Observe Changes in a View Hierarchy

In this section, you’ll complete the editing feature to keep the model’s list of reminders and the user interface updated when a user changes a reminder. To keep the reminder data in the views synced, you’ll modify the view controller so that it accepts a parameter that defines a set of behaviors you want to perform every time a reminder changes.
