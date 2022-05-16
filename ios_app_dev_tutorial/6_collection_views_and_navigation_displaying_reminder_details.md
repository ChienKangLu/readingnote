# Collection Views and Navigation — Displaying Reminder Details

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/displaying-reminder-details)

## Section 1 Create a Reminder View

You’ll follow a familiar pattern. Start by adding a view controller with a list view. Later in the tutorial, you’ll assign a data source to the list, and you’ll configure the list and the list view cells.

In this section, you’ll create the reminder detail view and include a reminder property whose details you’ll display.

- In Swift, classes and structures must assign an initial value to all stored properties by the time an instance is created

## Section 2 Create an Enumeration for Rows

In this section, you’ll add a row enumeration that you’ll use to specify the appropriate content and unique styling for each row.

You’ll add design elements, such as image symbols and font styling, to distinguish between key reminder details at a glance.

## Section 3 Set Up the Data Source

In this section, you’ll create a custom configuration for the list cells in the reminder detail view. You’ll access the default configuration of the list cell and apply the appropriate text and styling for each row. You’ll also create the data source for the reminder detail collection view.

## Section 4 Set Up a Snapshot

In this section, you’ll follow the pattern you used in configuring the list cells for the reminder list view. For the list of details, you’ll create a default snapshot, customize it, and apply it.

## Section 5 Display the Detail View

When the user taps a reminder in the reminder list view, Today pushes a new view onscreen that shows the details of that reminder. In the previous sections of this tutorial, you created the view that shows the details of a reminder. In this section, you’ll make the list view controller create a new instance of the detail view and inject the appropriate reminder dependency into that view.

## Section 6 Style the Navigation Bar

After completing the reminder detail view, you’ll update the style of the app to ensure the views work together functionally and visually.

In this section, you’ll modify the appearance of the navigation bar to match the styling of the rest of your app.
