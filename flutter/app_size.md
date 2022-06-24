# Measuring your app's size

The size analysis tool is invoked by passing the `--analyze-size` flag when building:

- `flutter build apk --analyze-size`
- `flutter build appbundle --analyze-size`
- `flutter build ios --analyze-size`
- `flutter build linux --analyze-size`
- `flutter build macos --analyze-size`
- `flutter build windows --analyze-size`

## Reducing app size

When building a release version of your app, consider using the `--split-debug-info` tag. This tag can dramatically reduce code size. For an example of using this tag, see [Obfuscating Dart code](https://docs.flutter.dev/deployment/obfuscate).

## Source

[Measuring your app's size | Flutter](https://docs.flutter.dev/perf/app-size#reducing-app-size)
