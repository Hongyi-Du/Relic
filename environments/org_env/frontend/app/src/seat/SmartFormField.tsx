// Smart form field that routes to appropriate input component based on field type
import { TaskSelector } from "./fields/TaskSelector";
import { MemberSelector } from "./fields/MemberSelector";
import { RichTextInput } from "./fields/RichTextInput";

type Props = {
  name: string;
  value: string;
  onChange: (value: string) => void;
  required?: boolean;
  context: {
    tasks?: any[];
    issues?: any[];
    members?: any[];
    channels?: any[];
  };
};

// Field type inference from field name
const inferFieldType = (name: string): string => {
  const lower = name.toLowerCase();

  if (lower.includes("task_id") || lower === "task") return "task";
  if (lower.includes("issue_id") || lower === "issue") return "issue";
  if (lower.includes("agent_id") || lower === "assignee" || lower === "owner") return "member";
  if (lower === "mentions") return "member_list";
  if (lower.includes("channel")) return "channel";
  if (lower === "text" || lower === "comment" || lower === "body" ||
      lower === "notes" || lower === "description" || lower === "summary" ||
      lower === "rationale" || lower === "reason") return "rich_text";
  if (lower === "priority") return "enum:low,medium,high,critical";
  if (lower === "severity") return "enum:minor,major,critical";
  if (lower === "status") return "enum";

  return "text";
};

// Get placeholder text based on field type
const getPlaceholder = (name: string, type: string): string => {
  if (type === "task") return "Search tasks by ID or title...";
  if (type === "issue") return "Search issues...";
  if (type === "member") return "Select a member...";
  if (type === "member_list") return "Type @ to mention members...";
  if (type === "channel") return "Select a channel...";
  if (type === "rich_text") return "Type @ to mention someone...";
  return name.replace(/_/g, " ");
};

export function SmartFormField({ name, value, onChange, required, context }: Props) {
  const fieldType = inferFieldType(name);
  const placeholder = getPlaceholder(name, fieldType);

  // Task selector
  if (fieldType === "task" && context.tasks) {
    return (
      <TaskSelector
        value={value}
        onChange={onChange}
        tasks={context.tasks}
        placeholder={placeholder}
      />
    );
  }

  // Issue selector (reuse task selector)
  if (fieldType === "issue" && context.issues) {
    return (
      <TaskSelector
        value={value}
        onChange={onChange}
        tasks={context.issues}
        placeholder={placeholder}
      />
    );
  }

  // Member selector
  if (fieldType === "member" && context.members) {
    return (
      <MemberSelector
        value={value}
        onChange={onChange}
        members={context.members}
        placeholder={placeholder}
      />
    );
  }

  // Member list (mentions)
  if (fieldType === "member_list" && context.members) {
    return (
      <MemberSelector
        value={value}
        onChange={onChange}
        members={context.members}
        placeholder={placeholder}
        multiple={true}
      />
    );
  }

  // Channel selector
  if (fieldType === "channel" && context.channels) {
    return (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select a channel...</option>
        {context.channels.map((ch: any) => (
          <option key={ch.id} value={ch.id}>
            #{ch.id.replace(/_/g, "-")}
          </option>
        ))}
      </select>
    );
  }

  // Rich text with @mention support
  if (fieldType === "rich_text" && context.members) {
    return (
      <RichTextInput
        value={value}
        onChange={onChange}
        members={context.members}
        placeholder={placeholder}
        rows={3}
      />
    );
  }

  // Enum selector
  if (fieldType.startsWith("enum:")) {
    const options = fieldType.split(":")[1].split(",");
    return (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select...</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt.replace(/_/g, " ")}
          </option>
        ))}
      </select>
    );
  }

  // Fallback: regular input
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  );
}
