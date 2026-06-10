#let split-slide(
  body,
  img-content,
  image-side: "right",
) = {
  let items = if image-side == "right" {
    (body, img-content)
  } else {
    (img-content, body)
  }

  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    ..items,
  )
}

#let full-slide(body) = {
  body
}
