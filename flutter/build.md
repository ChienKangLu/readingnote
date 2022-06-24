## Flutter's build modes

A quick summary for when to use which mode is as follows:

- Use [debug](https://docs.flutter.dev/testing/build-modes#debug) mode during development, when you want to use [hot reload](https://docs.flutter.dev/development/tools/hot-reload).
- Use [profile](https://docs.flutter.dev/testing/build-modes#profile) mode when you want to analyze performance.
- Use [release](https://docs.flutter.dev/testing/build-modes#release) mode when you are ready to release your app.

## Debug

```shell
flutter run
```

## Release

```shell
flutter run --release
```

You can compile to release mode for a specific target with `flutter build <target>`. For a list of supported targets, use `flutter help build`.

Produce an upload package for your application:

```shell
flutter build appbundle
```

## Profile

```shell
flutter run --profile
```

## Source

[Flutter's build modes | Flutter](https://docs.flutter.dev/testing/build-modes)
