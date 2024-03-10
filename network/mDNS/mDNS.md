# mDNS

mDNS/DNS-SD as described in [RFC 6763](https://tools.ietf.org/html/rfc6763 "https://tools.ietf.org/html/rfc6763"). DNS-SD stands for DNS-Based Service Discovery

- In mDNS, there is no central DNS server

- If you wish to query an IP whose hostname you are aware of, then you send a multicast message to all the devices in the network asking if any of them identify with the hostname. One of the devices will match the hostname that you are querying. It will then respond with its IP address (again via multicast, to all devices on the network). All the devices on the network can then update their local phonebook (mDNS cache), mapping the hostname with the local IP.

## Reference

[A beginner&#8217;s guide to mDNS and DNS-SD &#8211; iotespresso.com](https://iotespresso.com/a-beginners-guide-to-mdns-and-dns-sd/)

https://developer.android.com/develop/connectivity/wifi/use-nsd#kotlin

[GitHub - avahi/avahi: Avahi - Service Discovery for Linux using mDNS/DNS-SD -- compatible with Bonjour](https://github.com/avahi/avahi)

[samples/training/NsdChat/src/com/example/android/nsdchat - platform/development - Git at Google](https://android.googlesource.com/platform/development/+/master/samples/training/NsdChat/src/com/example/android/nsdchat)
