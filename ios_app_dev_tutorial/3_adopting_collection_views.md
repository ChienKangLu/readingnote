# Adopting Collection Views

## Source

[Apple Developer Documentation](https://developer.apple.com/tutorials/app-dev-training/adopting-collection-views)

Collection views manage ordered collections of data items and use customizable layouts to present them.

Adopting collection views helps you separate the concerns of data, layout, and presentation, resulting in an app that’s more robust and extensible.

## Displaying Collections

- `UICollectionView` class that efficiently displays related items as cells in a scrollable view.

- **Collection view**s flexibly organize **cell**s into **section**s. In a music app, for example, a collection view can display your music as a long list of songs or organize them into sections by popularity, genre, or mood.

- Modern collection views automatically animate changes to the state of their data

## Process

1. Create a data source for your collection view.

2. Implement a cell provider that configures your collection view’s cells.

3. Generate the current state of the data.

4. Display the data in the user interface.
