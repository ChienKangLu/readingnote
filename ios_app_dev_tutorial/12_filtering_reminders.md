# Filtering Reminders

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/filtering-reminders)

## Section 1 Create a List Style Enumeration

In this section, you’ll define a `ReminderListStyle` enumeration, with cases for each of the available styles. The enumeration contains a function that you’ll use to filter reminders for each style.

## Section 2 Filter Reminders by List Style

For users to more easily find their reminders in Today, you’ll implement filtering for the collection view by using the enumeration you created in the previous section. You’ll use higher-order functions, which take a closure parameter to customize their behavior.

## Section 3 Display a Segmented Control

`ReminderListViewController`displays reminders based on the specified list style. To allow users to select different list styles, you’ll add a segmented control that they can use to filter reminders by their due date.

## Section 4 Add Action to Segmented Control

Complete the list style feature by writing code to update the list of reminders when the user switches between list styles.
