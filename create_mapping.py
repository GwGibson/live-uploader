def generate_channel_mapping(start_number: int, num_channels: int) -> str:
    mappings = []
    for i in range(num_channels):
        sensor_number = start_number + i
        channel_number = i + 1
        mappings.append(f"{sensor_number}:{channel_number}")

    return ", ".join(mappings)


def main():
    result = generate_channel_mapping(1149, 428)
    # Probably better to copy to clipboard or create a file?
    # Ultimately, this should be done in a more user-friendly way
    print(result)


if __name__ == "__main__":
    main()
