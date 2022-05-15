# Collection Views and Navigation — Making Reminders Identifiable

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/making-reminders-identifiable)

## Introduction

A *diffable data source* stores a list of identifiers that represents the identities of the items in the collection view. In this tutorial, you’ll make the reminder model *identifiable* so that you can uniquely identify each reminder.

## Section 1 Make the Model Identifiable

You use identifiers to inform the data source of which items to include in the collection view and which items to reload when data changes. In the last tutorial, you used a reminder’s title as an identifier. Consider what would happen if a user were to change the title or create two reminders with the same title.

In this section, you’ll make the `Reminder` structure conform to the  `Identifiable` protocol. Then, you’ll update your code to use a reminder’s new `id` property when adding items to a data snapshot and configuring a collection view cell.

## Section 2 Create Functions for Accessing the Model

In this section, you’ll write functions that use a reminder’s identifier to retrieve and update individual items in the reminders array.

## Section 3 Create a Custom Button Action

Currently, users can view a reminder’s completion status, but they can’t change it. In this section, you’ll begin to make Today more interactive by writing code to update a reminder’s status when a user taps the done button.

## Section 4 Wire a Target-Action Pair

*Target-action* is a design pattern in which an object holds the information necessary to send a message to another object when an event occurs. In the Today app, the `touchUpInside` event occurs when a user taps the done button, which sends the `didPressDoneButton:sender` message to the view controller.

## Section 5 Update the Snapshot

When you work with diffable data sources, you apply a snapshot to update your user interface when data changes. In this section, you’ll create and apply a new snapshot when a user taps a done button.

## Section 6 Make the Action Accessible

A well-designed iOS app is accessible to all users, regardless of their abilities.

In this section, you’ll add an accessibility action to each cell so that users of VoiceOver and other assistive tools can mark a reminder as complete without tapping the done button. You’ll also add an accessibility value so that VoiceOver can read a reminder’s completion status to users.

## Section 7 Preview Using the Accessibility Inspector

You can use the Accessibility Inspector tool to identify and fix accessibility issues in your app, emulate VoiceOver, and even navigate through your user interface.

In this section, you’ll use the Accessibility Inspector to view a cell’s accessibility value and perform its accessibility action.
