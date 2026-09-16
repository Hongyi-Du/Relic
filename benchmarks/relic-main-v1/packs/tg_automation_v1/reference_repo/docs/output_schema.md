# Output Schema

All exported reports use snake_case field names. The contracts below are
stable across runs and are what downstream analytics should bind against.

## analyze_interactions

| field            | type     | description                                       |
| ---------------- | -------- | ------------------------------------------------- |
| `user_id`        | int      | Telegram user identifier                          |
| `username`       | string   | Telegram username (may be empty if not set)       |
| `messages_sent`  | int      | Messages authored by the user in the window       |
| `reactions_given`| int      | Reactions issued by the user in the window        |
| `last_active_at` | ISO 8601 | Timestamp of last activity (UTC)                  |
| `active_days`    | int      | Distinct days with any activity                   |
| `group_id`       | string   | Group identifier (added by the CLI)               |

## filter_users

Same columns as `analyze_interactions`.

## engagement (posts)

| field                | type   | description                            |
| -------------------- | ------ | -------------------------------------- |
| `group_id`           | string | Group identifier                       |
| `post_id`            | int    | Telegram post id                       |
| `author_id`          | int    | Post author identifier                 |
| `author_username`    | string | Post author username                   |
| `date`               | ISO 8601 | Post timestamp in UTC                 |
| `reactions_total`    | int    | Total reactions on the post            |
| `reactions_by_emoji` | dict   | Map emoji → count (stringified in CSV) |
| `replies`            | int    | Reply count                            |
| `comments`           | int    | Comment count                          |

## growth bulk-invite

| field      | type   | description                       |
| ---------- | ------ | --------------------------------- |
| `group_id` | string | Target group                      |
| `username` | string | Invited username                  |
| `status`   | string | `invited` / `skipped` / `error`   |
| `error`    | string | Error detail (when not `invited`) |

## growth trend

| field            | type   | description                                |
| ---------------- | ------ | ------------------------------------------ |
| `date`           | string | Day in ISO 8601 format, UTC                |
| `new_members`    | int    | Members who joined that day                |
| `active_members` | int    | Distinct members observed that day         |

## growth invite-link

| field         | type     | description                                   |
| ------------- | -------- | --------------------------------------------- |
| `group_id`    | string   | Target group                                  |
| `url`         | string   | Invite URL                                    |
| `expires_at`  | ISO 8601 | Expiration timestamp (empty when unset)       |
| `join_limit`  | int      | Maximum joins allowed (empty when unlimited)  |
