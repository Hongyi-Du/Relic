// Task selector with fuzzy search and status display
import { useState, useRef, useEffect } from "react";

type Task = {
  id: string;
  title: string;
  status: string;
  priority?: string;
  owner_id?: string;
};

type Props = {
  value: string;
  onChange: (value: string) => void;
  tasks: Task[];
  placeholder?: string;
};

export function TaskSelector({ value, onChange, tasks, placeholder }: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [focused, setFocused] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Find current task for display
  const current = tasks.find((t) => t.id === value);

  // Fuzzy filter tasks by search term
  const filtered = tasks.filter((t) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      t.id.toLowerCase().includes(q) ||
      t.title.toLowerCase().includes(q) ||
      t.status.toLowerCase().includes(q)
    );
  }).slice(0, 10); // Limit to 10 results

  useEffect(() => {
    if (open && filtered.length > 0) {
      setFocused(0);
    }
  }, [open, filtered.length]);

  const select = (task: Task) => {
    onChange(task.id);
    setSearch("");
    setOpen(false);
    inputRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocused((f) => Math.min(f + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocused((f) => Math.max(f - 1, 0));
    } else if (e.key === "Enter" && filtered.length > 0) {
      e.preventDefault();
      select(filtered[focused]);
    } else if (e.key === "Escape") {
      setOpen(false);
      setSearch("");
    }
  };

  return (
    <div className="smart-field task-selector">
      <input
        ref={inputRef}
        type="text"
        value={open ? search : (current ? `${current.id}: ${current.title}` : value)}
        onChange={(e) => {
          setSearch(e.target.value);
          if (!open) setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder || "Search tasks..."}
      />
      {open && filtered.length > 0 && (
        <div className="smart-field-dropdown">
          {filtered.map((task, i) => (
            <button
              key={task.id}
              className={`dropdown-item ${i === focused ? "focused" : ""}`}
              onClick={() => select(task)}
              onMouseEnter={() => setFocused(i)}
            >
              <div className="task-item">
                <span className="task-id">{task.id}</span>
                <span className="task-title">{task.title}</span>
                <span className={`task-status status-${task.status}`}>
                  {task.status.replace(/_/g, " ")}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
