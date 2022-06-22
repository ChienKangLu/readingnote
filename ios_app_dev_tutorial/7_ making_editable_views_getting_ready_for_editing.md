# Making Editable Views - Getting Ready for Editing

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/getting-ready-for-editing)

## Section 1 Create Sections for an Editing Mode

In future tutorials, you’ll add interactive controls that let users modify a reminder’s title, due date, and notes attributes. So that users can more easily interact with the controls, you’ll modify the data source to display each attribute in a separate section of the collection view when the view is in its editing mode.

## Section 2 Configure the View and Editing Modes

`ReminderViewController` is responsible for configuring the view for both editing and viewing reminder information. In this section, you’ll refactor the view controller to create separate viewing and editing snapshots. And you’ll start building separate cell configurations for the view and editing modes.

## Section 3 Add an Edit Button

In this section, you’ll add a button to the reminder view controller to enter and exit editing mode. You’ll update the data source snapshots when the view transitions between modes.

## Section 4 Show Headers in Editing Mode

In this section, you’ll update `ReminderViewController` to display headers in the collection view while in editing mode.

In UIKit, by default, collection views have no headers. You can add headers by including a supplementary data array with header titles. Or you can treat the first element in the data array as the header.
